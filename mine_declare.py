'''
Mine Declare constraints from the processed event logs.

    python mine_declare.py --dataset sepsis
    python mine_declare.py --dataset bpi12 --min_confidence 0.95 --prefix_safe_only
    python mine_declare.py --dataset sepsis --templates response precedence chainresponse
    python mine_declare.py --dataset sepsis --miner tensor
    python mine_declare.py --dataset sepsis --sort_by class_gap confidence --top_k_per_template 5 --top_k 30
    python mine_declare.py --dataset bpi12 --max_ltn_gap 0.01 --redundancy_tolerance 0.02 --exclude_templates nonsuccession

By default constraints are discovered with pm4py (declare.PM4PyDeclareMiner, pm4py template names
for --templates); --miner tensor uses the torch implementation of the LTN templates
(declare.DeclareMiner, declare.templates names for --templates).
For each constraint: supp (fraction of traces not violating it), act (fraction of traces activating
it), conf (satisfied / activated), conf+ / conf- (conf on positive / negative traces) and, for pm4py,
ltn_conf (conf under the semantics of the LTN predicates, which can differ from pm4py's).

Constraints are mined on the training cases only: the test split is reproduced with the same
seeded create_test_set used by data/preprocess_<dataset>.py. Traces are truncated to the
model's max prefix length so that the mined knowledge describes what the models see.
After mining, the constraints are pruned and ordered with declare.prune_constraints (thresholds,
template / activity selection, redundancy and transitive reduction, top-k) and sorted by --sort_by
(metrics in declare.METRICS, descending unless suffixed with ":asc").
The result is written to results/declare_<dataset>.json and can be loaded as LTN predicates with
declare.DeclareKB.from_json(path, feature_names, max_prefix_length).
'''
import argparse
import contextlib
import io

import pandas as pd

from data import preprocess_bpi12, preprocess_bpi17, preprocess_sepsis, preprocess_traffic
from declare import DeclareMiner, PM4PyDeclareMiner, PM4PY_TEMPLATES, activity_vocab, encode_log, TEMPLATES, \
    DEFAULT_TEMPLATES, METRICS, prune_constraints

DATASETS = {
    "sepsis": {"path": "data_processed/sepsis_3.csv", "module": preprocess_sepsis, "max_prefix_length": 13},
    "bpi12": {"path": "data_processed/bpi12.csv", "module": preprocess_bpi12, "max_prefix_length": 40},
    "bpi17": {"path": "data_processed/bpi17.csv", "module": preprocess_bpi17, "max_prefix_length": 20},
    "traffic": {"path": "data_processed/traffic_fines.csv", "module": preprocess_traffic, "max_prefix_length": 10},
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sepsis", choices=DATASETS)
    parser.add_argument("--miner", type=str, default="pm4py", choices=["pm4py", "tensor"])
    parser.add_argument("--templates", type=str, nargs="+", default=None,
                        help=f"pm4py: {PM4PY_TEMPLATES} (default: all); tensor: {list(TEMPLATES)} "
                             f"(default: {DEFAULT_TEMPLATES})")
    parser.add_argument("--min_support", type=float, default=0.05, help="Minimum activation support")
    parser.add_argument("--min_confidence", type=float, default=0.9, help="Minimum confidence")
    parser.add_argument("--min_activity_support", type=float, default=0.01,
                        help="Skip activities occurring in fewer traces than this fraction")
    parser.add_argument("--prefix_safe_only", action="store_true",
                        help="Only mine templates that can be soundly enforced on prefixes")
    # Pruning
    parser.add_argument("--min_ltn_confidence", type=float, default=None,
                        help="Minimum confidence under the LTN predicate semantics")
    parser.add_argument("--max_ltn_gap", type=float, default=None,
                        help="Drop constraints whose confidence differs by more than this from the confidence "
                             "under the LTN predicate semantics (|conf - ltn_conf|, pm4py only)")
    parser.add_argument("--min_class_gap", type=float, default=None,
                        help="Keep constraints with |conf+ - conf-| >= this (label-discriminative constraints)")
    parser.add_argument("--exclude_templates", type=str, nargs="+", default=None,
                        help="Templates to drop (LTN or pm4py names)")
    parser.add_argument("--activities", type=str, nargs="+", default=None,
                        help="Keep only constraints whose activities are all in this list")
    parser.add_argument("--no_pruning", action="store_true",
                        help="Disable redundancy and transitive reduction")
    parser.add_argument("--no_transitive", action="store_true",
                        help="Disable the transitive reduction of Response/Precedence/Succession/RespondedExistence")
    parser.add_argument("--redundancy_tolerance", type=float, default=None,
                        help="Drop an implied constraint only if the implying one has confidence >= its "
                             "confidence - tolerance (default: always drop)")
    parser.add_argument("--top_k_per_template", type=int, default=None)
    parser.add_argument("--top_k", "--max_constraints", dest="top_k", type=int, default=None)
    # Ordering
    parser.add_argument("--sort_by", type=str, nargs="+", default=["confidence", "activation_support"],
                        help=f"Metrics to sort by, in priority order, optionally with ':asc' / ':desc'. "
                             f"Available: {list(METRICS)}")
    parser.add_argument("--full_traces", action="store_true", help="Do not truncate traces to the max prefix length")
    parser.add_argument("--include_test", action="store_true", help="Mine on the whole log (leaks test cases)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    valid = PM4PY_TEMPLATES if args.miner == "pm4py" else list(TEMPLATES)
    unknown = set(args.templates or []) - set(valid)
    if unknown:
        parser.error(f"unknown templates for --miner {args.miner}: {sorted(unknown)}; choose from {valid}")
    for key in args.sort_by:
        name, _, order = key.partition(":")
        if name not in METRICS or order not in ("", "asc", "desc"):
            parser.error(f"invalid --sort_by '{key}': use one of {list(METRICS)}, optionally with ':asc' / ':desc'")
    return args


def main():
    args = get_args()
    cfg = DATASETS[args.dataset]

    print("-- Reading dataset")
    data = pd.read_csv(cfg["path"], dtype={"org:resource": str})
    data = data.dropna(subset=["case:concept:name"])
    # Same activity codes as the preprocessing (pd.Categorical over the whole log)
    vocab = activity_vocab(data["concept:name"])

    if not args.include_test:
        with contextlib.redirect_stdout(io.StringIO()):
            train_ids, test_ids = cfg["module"].create_test_set(data)
        data = data[data["case:concept:name"].isin(train_ids)]
        print(f"Mining on {len(train_ids)} training cases ({len(test_ids)} test cases excluded)")

    max_len = None if args.full_traces else cfg["max_prefix_length"]
    traces, _, labels = encode_log(data, vocab, max_len=max_len)

    miner_cls = PM4PyDeclareMiner if args.miner == "pm4py" else DeclareMiner
    miner = miner_cls(
        templates=args.templates,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_activity_support=args.min_activity_support,
        prefix_safe_only=args.prefix_safe_only,
        prune_redundant=False,
    )
    mined = miner.fit(traces, labels, vocab=vocab)
    selection = dict(
        min_ltn_confidence=args.min_ltn_confidence, max_ltn_gap=args.max_ltn_gap, min_class_gap=args.min_class_gap,
        exclude_templates=args.exclude_templates, activities=args.activities,
        redundant=not args.no_pruning, transitive=not (args.no_pruning or args.no_transitive),
        redundancy_tolerance=args.redundancy_tolerance, sort_by=args.sort_by,
        top_k_per_template=args.top_k_per_template, top_k=args.top_k)
    constraints = prune_constraints(mined, **selection)
    miner.constraints = constraints

    print(f"-- {len(mined)} Declare constraints mined with {args.miner} (traces: {len(traces)}, "
          f"activities: {len(vocab)}), {len(constraints)} after pruning, sorted by {args.sort_by}")
    for c in constraints:
        print(("  [prefix-safe] " if c.spec.prefix_safe else "  ") + str(c))

    output = args.output or f"results/declare_{args.dataset}.json"
    miner.save(output, vocab=vocab, dataset=args.dataset, miner=args.miner, max_len=max_len,
               train_only=not args.include_test, selection=selection)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()

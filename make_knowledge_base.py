'''
Generate knowledge_base.txt: the knowledge injected in the models, with statistics on the event logs.

    python make_knowledge_base.py                      # all datasets
    python make_knowledge_base.py --datasets sepsis    # a subset (writes only those datasets)

For each dataset it lists
    - the logical rules (nesy/knowledge.py), used by feat_eng, semantic, A and B
    - the predefined Declare (control-flow) constraints (C with --declare_source predefined)
    - the best Declare constraints mined from the training prefixes (C, default), selected as in main.py
with statistics on the entire traces of the train, val and test cases and of the whole event log.
'''
import argparse
import contextlib
import dataclasses
import io
import time

import torch

from declare import activity_slice, activity_vocab
from declare.templates import TEMPLATES
from nesy.datasets import DATASETS, encode_traces, load_dataset
from nesy.declare_knowledge import (PREDEFINED_CONSTRAINTS, add_declare_args, constraint_stats,
                                    mine_declare_knowledge)
from nesy.knowledge import build_knowledge

LABELS = {
    "bpi12": "the loan application is accepted",
    "bpi17": "the loan offer is accepted",
    "sepsis": "the patient is admitted to the ICU",
    "traffic": "the fine is sent for credit collection",
}
SPLITS = ["train", "val", "test", "log"]
WIDTH = 11


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--output", default="knowledge_base.txt")
    add_declare_args(parser)
    return parser.parse_args()


def row(title, values, fmt):
    cells = "".join(f"{'-' if v is None else fmt.format(v):>{WIDTH}}" for v in values)
    return f"      {title:<28}{cells}"


def header():
    return f"      {'traces of':<28}" + "".join(f"{s:>{WIDTH}}" for s in SPLITS)


def pct(value):
    return None if value is None else 100 * value


def dataset_section(data, args):
    traces = encode_traces(data)
    masks = {s: torch.from_numpy(traces.split == s) for s in ("train", "val", "test")}
    masks["log"] = torch.ones(len(traces.y), dtype=torch.bool)  # every case
    n = {s: int(masks[s].sum()) for s in SPLITS}
    lengths = (traces.layout.get(traces.x, "concept:name") != 0).sum(dim=1)

    out = ["=" * 100, f"# {data.name.upper()}  ({DATASETS[data.name].path})", "=" * 100,
           f"Label 1: {LABELS[data.name]}.",
           f"Event log: {n['log']} cases, {len(data.events)} events, {data.events['concept:name'].nunique()} activities; "
           f"trace length min {int(lengths.min())}, mean {float(lengths.float().mean()):.1f}, max {int(lengths.max())}. "
           f"The models see prefixes of at most {data.layout.seq_len} events.",
           header(),
           row("cases", [n[s] for s in SPLITS], "{:d}"),
           row("label 1 (%)", [pct(float(traces.y[masks[s]].mean())) for s in SPLITS], "{:.1f}")]

    # Logical rules, evaluated on the entire traces
    knowledge = build_knowledge(dataclasses.replace(data, layout=traces.layout))
    out += ["", "## Logical rules (nesy/knowledge.py): inputs of P (feat_eng, A), semantic loss, axioms on P (B)",
            "   activated: traces satisfying the condition; support: traces satisfying the rule (condition -> label);",
            "   confidence: activated traces with the implied label"]
    for rule in knowledge.rules:
        fired = rule.condition(traces.x)
        holds = ~fired | (traces.y == rule.label)
        note = "   [constraint only: not an input of P, not counted in the compliance]" if rule.constraint_only else ""
        out += ["", f"   {rule.name}{note}", f"      {rule.formula}", header()]
        out.append(row("activated (%)", [pct(float(fired[masks[s]].float().mean())) for s in SPLITS], "{:.1f}"))
        out.append(row("support (%)", [pct(float(holds[masks[s]].float().mean())) for s in SPLITS], "{:.1f}"))
        out.append(row("confidence (%)", [pct(float(holds[masks[s] & fired].float().mean()))
                                          if (masks[s] & fired).any() else None for s in SPLITS], "{:.1f}"))

    # Declare constraints, evaluated on the activity sequences of the entire traces
    acts = activity_slice(traces.x, list(traces.layout.feature_names), traces.layout.seq_len)
    labels = traces.y.round().long()
    out += ["", "## Declare constraints (nesy/declare_knowledge.py), never connected to the label: inputs of P in C",
            "   (--declare_in_p, default) or axioms Forall x: C(x) of the knowledge base (--no-declare_in_p)",
            "   support: traces satisfying the constraint (vacuously included); activated: traces activating it;",
            "   confidence: activated traces satisfying it; on label 1 / label 0: the same on the traces of each label"]

    out += ["", "### Predefined (python main.py --declare_source predefined)"]
    vocab = activity_vocab(data.raw["concept:name"])
    for rule_id, template, a, b, description in PREDEFINED_CONSTRAINTS[data.name]:
        out += declare_rows(f"{rule_id}: {display(template, a, b)}  ({description})", template, vocab[a], vocab[b],
                            acts, labels, masks)
    if data.name == "bpi12":
        out += ["", "   R5: Response(W_Valideren aanvraag-COMPLETE, O_ACCEPTED-COMPLETE) cannot be grounded: "
                    "O_ACCEPTED defines the label and is not in the processed log"]
    if data.name == "traffic":
        out += ["", "   R6 named the target 'Payment Notification', which is not an activity of the log: "
                    "'Payment' is used"]

    with contextlib.redirect_stdout(io.StringIO()):
        mined = mine_declare_knowledge(data, args)
    out += ["", f"### Mined (python main.py, default): the best {len(mined.constraints)} constraints",
            f"   mined with pm4py on the training prefixes: activation support >= {args.declare_min_support}, "
            f"confidence >= {args.declare_min_confidence},",
            f"   |conf - ltn_conf| <= {args.declare_max_ltn_gap}, prefix-safe templates, no redundant / transitive / "
            f"duplicate constraints;",
            f"   ranked by class gap |conf+ - conf-| on the training prefixes, top {args.declare_top_k}"]
    for k, c in enumerate(mined.constraints):
        gap = abs(c.confidence_pos - c.confidence_neg) if None not in (c.confidence_pos, c.confidence_neg) else None
        title = f"D{k + 1}: {c.name}" + (f"  (class gap on the training prefixes {gap:.3f})" if gap is not None else "")
        out += declare_rows(title, c.template, c.a, c.b, acts, labels, masks)
    out.append("")
    return out


def display(template, a, b):
    return f"{TEMPLATES[template].display_name}({a}, {b})"


def declare_rows(title, template, a, b, acts, labels, masks):
    stats = {s: constraint_stats(template, a, b, acts[masks[s]], labels[masks[s]]) for s in SPLITS}
    safe = "prefix-safe" if TEMPLATES[template].prefix_safe else "not prefix-safe"
    return ["", f"   {title}   [{safe}]", header(),
            row("support (%)", [pct(stats[s]["support"]) for s in SPLITS], "{:.1f}"),
            row("activated (%)", [pct(stats[s]["activated"]) for s in SPLITS], "{:.1f}"),
            row("confidence (%)", [pct(stats[s]["confidence"]) for s in SPLITS], "{:.1f}"),
            row("confidence on label 1 (%)", [pct(stats[s]["confidence_pos"]) for s in SPLITS], "{:.1f}"),
            row("confidence on label 0 (%)", [pct(stats[s]["confidence_neg"]) for s in SPLITS], "{:.1f}")]


def main():
    args = get_args()
    lines = [
        "'''",
        "Knowledge injected in the models, with statistics on the event logs.",
        f"Generated by make_knowledge_base.py on {time.strftime('%Y-%m-%d')}: do not edit by hand, the rules are",
        "defined in nesy/knowledge.py (logical rules) and nesy/declare_knowledge.py (Declare constraints).",
        "",
        "Statistics are computed on the entire traces (all the events of each case) of the train / val / test",
        "cases of the fixed split of data/preprocess_<dataset>.py, and of the whole event log ('log').",
        "P(x) is the label predicate (label 1).",
        "'''",
        "",
    ]
    for name in args.datasets:
        print(f"-- {name}")
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_dataset(name)
        lines.extend(dataset_section(data, args))
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"-- Written {args.output}")


if __name__ == "__main__":
    main()

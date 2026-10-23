'''
Create the rule columns rule_1, rule_2, rule_3 of the processed event logs.

    python make_rule_labels.py                              # all datasets, written to data_processed/rule_labels/
    python make_rule_labels.py --datasets sepsis bpi12      # a subset
    python make_rule_labels.py --datasets bpi12 --inplace   # overwrite data_processed/bpi12.csv (backup: .bak)

rule_k is 1 on every event of a case whose entire trace satisfies the antecedent of the logical rule Rk of
nesy/knowledge.py (R1, R2, R3: the rules used as input features and in the compliance), 0 otherwise.
The consequent (the label implied by the rule) is not considered.

The columns matter: data/preprocess_<dataset>.create_test_set builds the test set from the rule-compliant
cases, so changing them changes the train / val / test split and every result. That is why the new files are
written to --output_dir by default; with --inplace the original is first copied to <file>.bak. All the other
columns are written back exactly as read.
'''
import argparse
import contextlib
import dataclasses
import io
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from nesy.datasets import DATASETS, FeatureLayout, PreparedData, encode_traces
from nesy.knowledge import build_knowledge

RULE_COLUMNS = ["rule_1", "rule_2", "rule_3"]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--output_dir", default="data_processed/rule_labels")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the processed logs (after a .bak copy)")
    return parser.parse_args()


@contextlib.contextmanager
def encoding_only(module):
    '''
    Run module.preprocess_eventlog for its encoding (categorical codes, scalers) only. The encoding is fitted on the
    whole log and does not depend on the split, so the split is skipped (every case in "train") and the prefixes
    are built for one case only, which is enough to know the feature names.
    '''
    captured = {}
    original_split, original_ngrams = module.create_test_set, module.create_ngrams

    def create_test_set(data, *args, **kwargs):
        return list(data["case:concept:name"].unique()), []

    def create_ngrams(data, train_ids, val_ids, test_ids, *args, **kwargs):
        one_case = data["case:concept:name"].iloc[:1].tolist()
        *_, feature_names = original_ngrams(data, one_case, [], [], *args, **kwargs)
        captured.update(events=data, feature_names=feature_names, case_ids=list(train_ids))
        return None

    module.create_test_set, module.create_ngrams = create_test_set, create_ngrams
    try:
        yield captured
    finally:
        module.create_test_set, module.create_ngrams = original_split, original_ngrams


def rule_labels(name, raw):
    '''(case ids, (N, 3) bool): truth of the antecedents of R1-R3 on the entire trace of each case.'''
    module = DATASETS[name].preprocessing
    raw = raw.copy()
    for column in RULE_COLUMNS:  # create_ngrams drops them
        if column not in raw:
            raw[column] = 0
    # In a temporary directory: some preprocessing scripts write the (here empty) test ids to the working directory
    with encoding_only(module) as log, contextlib.redirect_stdout(io.StringIO()), \
            tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        _, vocab_sizes, scalers = module.preprocess_eventlog(raw)
    layout = FeatureLayout(tuple(log["feature_names"]), seq_len=1)  # seq_len is set by encode_traces
    data = PreparedData(name, raw, None, None, None, layout, vocab_sizes, scalers, log["events"],
                        {"train": log["case_ids"], "val": [], "test": []})
    traces = encode_traces(data)
    knowledge = build_knowledge(dataclasses.replace(data, layout=traces.layout))
    assert [r.name.split(":")[0] for r in knowledge.feature_rules] == ["R1", "R2", "R3"]
    return traces.case_ids, knowledge.truth(traces.x).numpy()


def main():
    args = get_args()
    if not args.inplace:
        os.makedirs(args.output_dir, exist_ok=True)
    for name in args.datasets:
        path = DATASETS[name].path
        print(f"-- {name}: {path}")
        raw = pd.read_csv(path, dtype={"org:resource": str})
        text = pd.read_csv(path, dtype=str, keep_default_na=False)  # the columns as written, to keep them unchanged
        case_ids, truth = rule_labels(name, raw)
        values = pd.DataFrame(truth.astype(int), index=case_ids, columns=RULE_COLUMNS)

        cases = raw["case:concept:name"]
        has_case = cases.notna().to_numpy()
        for column in RULE_COLUMNS:
            new = np.zeros(len(raw), dtype=int)  # events without a case id (dropped by the preprocessing): 0
            new[has_case] = values[column].reindex(cases[has_case]).fillna(0).astype(int).to_numpy()
            if column in raw:
                old_cases = raw[has_case].groupby("case:concept:name")[column].first()
                new_cases = values[column].reindex(old_cases.index).fillna(0)
                changed = int((old_cases.round() != new_cases).sum())
                print(f"   {column}: {int(new_cases.sum())} cases (before {int(old_cases.sum())}), "
                      f"{changed} of {len(old_cases)} cases changed")
            else:
                print(f"   {column}: {int(values[column].sum())} of {len(values)} cases (new column)")
            text[column] = new

        output = path if args.inplace else os.path.join(args.output_dir, os.path.basename(path))
        if args.inplace and not os.path.exists(path + ".bak"):
            shutil.copy2(path, path + ".bak")
            print(f"   original copied to {path}.bak")
        text.to_csv(output, index=False)
        print(f"   written {output}")


if __name__ == "__main__":
    main()

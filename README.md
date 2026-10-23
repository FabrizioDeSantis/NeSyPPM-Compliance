# NeSyPM-Compliance

Neuro-symbolic **outcome prediction** for business processes with **compliance knowledge**.

Given the prefix of a running case (its first events), a neural model predicts the outcome of the case:
whether a loan is accepted, a patient goes to the ICU, or a fine is sent for credit collection. Domain
knowledge is injected into the model with [Logic Tensor Networks](https://github.com/tommasocarraro/LTNtorch)
(LTN) or with simpler baselines. Two kinds of knowledge are used:

- **logical rules** on the data and the label, e.g. *patients with lactic acid > 2 should be admitted to the ICU*;
- **Declare (control-flow) constraints** on the order of the activities, e.g. *ER Sepsis Triage immediately
  follows ER Triage*. They are mined from the training log or hand-written.

The models are evaluated on predictive performance and on **compliance**: how often the predictions agree with
the rules.

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Configurations](#configurations)
- [Knowledge](#knowledge)
- [Experimental protocol](#experimental-protocol)
- [Command-line options](#command-line-options)
- [Outputs](#outputs)
- [Repository structure](#repository-structure)
- [Extending](#extending)

## Setup

```bash
conda create -n nesypm python=3.11
conda activate nesypm
pip install -r requirements.txt     # pick the torch build matching your CUDA version
```

The processed event logs must be in `data_processed/` (see [Datasets](#datasets)). A GPU is used when one is
available.

## Quick start

```bash
# Every configuration on Sepsis, LSTM backbone, seed 42
python main.py --dataset sepsis

# A subset of the configurations, transformer backbone, three seeds (mean ± std is reported)
python main.py --dataset bpi12 --model transformer --configs vanilla ltn B ABC --seeds 0 1 2

# Control-flow knowledge: the hand-written Declare constraints instead of the mined ones
python main.py --dataset traffic --configs C BC ABC --declare_source predefined

# Control-flow knowledge kept out of P: axioms Forall x: C(x) instead of inputs of P
python main.py --dataset bpi17 --configs C ABC --no-declare_in_p

# Regenerate knowledge_base.txt (all the rules, with statistics on the event logs)
python make_knowledge_base.py
```

`python main.py --help` lists every option.

## Datasets

| `--dataset` | File | Label 1 | Cases | Max prefix length |
|---|---|---|---|---|
| `bpi12` | `data_processed/bpi12.csv` | the loan application is accepted | 4,804 | 40 |
| `bpi17` | `data_processed/bpi17.csv` | the loan offer is accepted | 31,417 | 20 |
| `sepsis` | `data_processed/sepsis_3.csv` | the patient is admitted to the ICU | 784 | 13 |
| `traffic` | `data_processed/traffic_fines.csv` | the fine is sent for credit collection | 150,370 | 10 |

`data/preprocess_<dataset>.py` encodes each log and builds the inputs:
- **Encoding:** categorical attributes become integer codes (0 is padding); numerical attributes are min-max scaled.
- **Split:** train / validation / test cases. The test set is built on purpose from cases that comply with the
  rules, so the rules hold far more often on test than on train.
- **Prefixes:** every prefix of every case (lengths 1 to n-1), truncated and padded to the max prefix length. A
  prefix is stored as one flat vector, feature by feature.

The split is seeded and deterministic.

The columns `rule_1`, `rule_2`, `rule_3` of the processed logs mark, for every event of a case, whether the entire
trace of the case satisfies the antecedent of rule R1, R2 or R3 (see [Knowledge](#knowledge)); the consequent (the
label implied by the rule) is not considered. They decide which
cases can go to the test set. `make_rule_labels.py` computes them from the rules of `nesy/knowledge.py`:

```bash
python make_rule_labels.py                             # all datasets, written to data_processed/rule_labels/
python make_rule_labels.py --datasets bpi12 --inplace  # overwrite data_processed/bpi12.csv (original kept as .bak)
```

All the other columns are written back unchanged. Changing the rule columns changes the train / val / test split,
and therefore every result: overwrite the logs only on purpose.

## Configurations

`--configs` selects what to train and test (default: every configuration except `C`).

| Configuration | Training | Knowledge |
|---|---|---|
| `vanilla` | binary cross-entropy | none |
| `feat_eng` | binary cross-entropy | truth values of the logical rules as extra inputs |
| `semantic` | cross-entropy + semantic loss | the logical rules |
| `ltn` | LTN, data axioms only | none |
| `A` | LTN | truth values of the logical rules as extra inputs of P |
| `B` | LTN | logical rules as axioms on P |
| `C` | LTN | Declare constraints (see below) |
| `AB`, `AC`, `BC`, `ABC` | LTN | the corresponding combinations |

The neural predicate P(x), the probability of label 1, is an LSTM (`--model lstm`, default) or a transformer
(`--model transformer`). The LTN loss is `1 - SatAgg(axioms)`, with:
- the **data axioms** `Forall x+: P(x+)` and `Forall x-: Not(P(x-))`;
- in **B**, one guarded axiom per logical rule, e.g. `Forall x with LacticAcid(x) > 2: P(x)`.

Declare constraints are never connected to the label. In the `C` configurations:
- `--declare_in_p` (default): their truth values are extra inputs of P, i.e. P(x, C(x)).
- `--no-declare_in_p`: they stay out of P and are axioms `Forall x: C(x)` of the knowledge base. P does not
  appear in these axioms, so they do not change its gradients.

## Knowledge

[`knowledge_base.txt`](knowledge_base.txt) lists all the knowledge, with statistics (support, activation and
confidence, by label) on the entire traces of the train, validation and test cases and of the whole log. It is
generated by `make_knowledge_base.py` from the code, so it always matches what the models are trained with.

**Logical rules** are defined once, by feature name, in [`nesy/knowledge.py`](nesy/knowledge.py):

| Dataset | Rules |
|---|---|
| BPI12 | R1 amount < 10k → not accepted · R2 40k < amount ≤ 60k → not accepted · R3 resource 11169 or 10910 involved → not accepted |
| BPI17 | R1 amount < 20k, offer created with credit score → accepted · R2 offer created without credit score → not accepted · R3 amount > 20k, existing loan takeover → not accepted |
| Sepsis | R1 lactic acid > 2 → ICU · R2 infection suspected, SIRS tachypnea and heart rate → ICU · R3 CRP before IV antibiotics and CRP > 100 → ICU · R7 ER triage and IV antibiotics within 120 → no ICU (only in the semantic loss and the B axioms) |
| Traffic | R1 paid less than the fine → credit collection · R2 penalty added → credit collection · R3 amount > 400 → credit collection |

**Declare constraints** are defined in [`nesy/declare_knowledge.py`](nesy/declare_knowledge.py). There are two
sources (`--declare_source`):

- `mined` (default): the best constraints mined from the training prefixes.
  1. They are mined with pm4py.
  2. The candidates are filtered:
     - only prefix-safe templates are kept;
     - confidence must be ≥ 0.9;
     - the confidence must be the same under pm4py and under the LTN semantics;
     - redundant, transitively implied and duplicate constraints are removed.
  3. The rest are ranked by class gap |conf(label 1) − conf(label 0)|, and the top 10 are kept.
- `predefined`: the hand-written constraints R4–R6 (`Response` / `ChainResponse`), e.g.
  `ChainResponse(ER Triage, ER Sepsis Triage)`.

## Experimental protocol

- **Same starting point:** for every seed (`--seeds`), each configuration starts from the same initialization and
  sees the batches in the same order.
- **Model selection:** after each epoch the model is scored on the validation set. The parameters with the best
  validation macro-F1 are kept, and training stops early after `--patience` epochs without improvement. The test
  set is used once, at the end.
- **Determinism:** Python, NumPy and PyTorch are seeded, cuDNN is deterministic, and the train / validation / test
  split does not depend on Python's hash seed. Two runs with the same arguments give the same numbers.
- **Metrics:**
  - accuracy, and macro-averaged F1, precision and recall;
  - **compliance**: consider the (rule, test prefix) pairs where a logical rule fires and the true label agrees
    with it. Compliance is the fraction of those pairs where the prediction agrees too. It is reported overall
    and per rule.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--dataset` | required | `bpi12`, `bpi17`, `sepsis`, `traffic` |
| `--model` | `lstm` | backbone of P: `lstm` or `transformer` |
| `--configs` | all except `C` | configurations to run, or `all` |
| `--seeds` | `42` | one run per seed |
| `--epochs` / `--epochs_ltn` | 15 / per dataset | maximum epochs of the non-LTN / LTN configurations (LTN: 5 on BPI12 and BPI17, 15 otherwise) |
| `--patience` | 5 | early stopping on the validation macro-F1 |
| `--batch_size`, `--lr` | 32, 1e-3 | |
| `--lambda_sem` | 0.1 | weight of the semantic loss |
| `--hidden_size`, `--model_dim`, `--num_heads`, `--num_layers`, `--dropout` | 128, 64, 4, 2, 0.1 | network sizes |
| `--declare_source` | `mined` | `mined` or `predefined` Declare constraints |
| `--declare_in_p` / `--no-declare_in_p` | in P | Declare constraints as inputs of P or as axioms `Forall x: C(x)` |
| `--declare_top_k`, `--declare_min_support`, `--declare_min_confidence`, `--declare_max_ltn_gap` | 10, 0.05, 0.9, 0.01 | selection of the mined constraints |
| `--output_dir` | `results` | where results are written |

## Outputs

- `results/runs/<dataset>_<model>_<timestamp>.json`: arguments, library versions, rules, Declare constraints, and
  every metric (including per-rule compliance and confusion matrices) for each seed and configuration.
- `results/results.csv`: one row per (run, seed, configuration), appended. Convenient for building tables.
- `results/declare_<dataset>_<source>.json`: the Declare constraints used by the C configurations.

## Repository structure

```
main.py                    entry point: train and test the configurations
make_knowledge_base.py     generates knowledge_base.txt
make_rule_labels.py        computes the rule_1..rule_3 columns of the processed logs
knowledge_base.txt         the knowledge, with statistics on the event logs
nesy/
  datasets.py              dataset registry, loading, entire-trace encoding, data loaders
  knowledge.py             logical rules of each dataset
  declare_knowledge.py     Declare constraints: mined (default) or predefined
  training.py              configurations, seeding, models, losses (BCE, semantic, LTN), training loop
  evaluation.py            test metrics and compliance
model/
  inputs.py                flat prefix -> per-event vectors (embeddings, numerical features, extra inputs)
  lstm.py, transformer.py  backbones of P
  semantic_loss.py
data/
  preprocess_<dataset>.py  encoding, train/val/test split, prefixes
  dataset.py               PyTorch dataset of prefixes
declare/                   Declare templates, miners (pm4py and tensor-based), LTN predicates
mine_declare.py            stand-alone Declare mining on a dataset
demo_declare_kb.py         toy example of Declare constraints as LTN predicates
legacy/                    the previous per-dataset scripts, kept for reference (see legacy/README.md)
```

## Extending

- **A new rule:** add a `Rule` to the dataset's function in `nesy/knowledge.py`. It is then used as an input
  feature, in the semantic loss, as a B axiom and in the compliance metric. Regenerate `knowledge_base.txt`
  (and, to use a changed R1–R3 for the split, the rule columns with `make_rule_labels.py`).
- **A new predefined Declare constraint:** add it to `PREDEFINED_CONSTRAINTS` in `nesy/declare_knowledge.py`. The
  template names are listed in `declare/templates.py`.
- **A new dataset:** write `data/preprocess_<name>.py` with the same `preprocess_eventlog` interface, register it
  in `DATASETS` (`nesy/datasets.py`), and add its rules to `RULES` (`nesy/knowledge.py`).

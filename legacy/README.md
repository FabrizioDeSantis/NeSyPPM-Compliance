# Legacy experiment scripts

These are the `main_<dataset>_<approach>.py` scripts that `main.py` replaced, kept for reference along with the
model / dataset / metrics code they import (`legacy/model`, `legacy/data/dataset.py`, `legacy/metrics.py`).
Run them from the repository root with `PYTHONPATH=. python legacy/<script>.py`.

| Old script | New command |
|---|---|
| `main_<dataset>_compliance.py` | `python main.py --dataset <dataset> --configs vanilla ltn B A AB BC AC ABC` |
| `main_traffic_compliance_transformer.py` | `python main.py --dataset traffic --model transformer --configs vanilla ltn B A AB BC AC ABC` |
| `main_<dataset>_feat_eng.py` | `python main.py --dataset <dataset> --configs feat_eng` |
| `main_<dataset>_semantic(_loss).py` | `python main.py --dataset <dataset> --configs semantic` |
| `main_sepsis_compliance_declare.py` | `python main.py --dataset sepsis` |

## Why the results differ from `main.py`

Issues in these scripts that `main.py` fixes:

- **Rules defined many times, inconsistently.** Each script (and `metrics.compute_metrics` / `compute_metrics_fa`)
  had its own copy of the rules, with hardcoded column indices. Examples: BPI12 R3 combined the two resources with
  AND for training but OR for evaluation; the sepsis lactic acid threshold was 2 in training and 4 in
  `compute_metrics`; `compute_metrics_fa` ignored sepsis R1 and checked R3 against label 0. So the compliance values
  of different configurations were not comparable.
- **Wrong column indices for the current data layout:**
  - BPI12: the input has 3 features x 40 events = 120 columns, but the amount rules read `x[:, 200:240]`, an empty
    slice, so R1 and R2 never fired. The resource rules scanned every column, so `== 21` also matched activity code 21.
  - The "A" models (`LSTMModelA`, `EventTransformerA`) read the appended rule features at per-dataset hardcoded
    offsets. With `sepsis_3.csv` (30 features) these offsets pointed at `time_since_previous` and `elapsed_time`, so
    only one of the three rules reached the model.
  - The C predicates (`HasAct`, `IsNext`) read columns `104:117`, which is the sepsis activity slice, on BPI12, BPI17
    and Traffic too. The sepsis ER Triage / ER Sepsis Triage codes were swapped.
- **Knowledge with no effect on training.** The "C" axioms had the same formula as body and guard, so they were
  always true. The "A" axioms (`Forall x: Rule(x)`) contain no learnable predicate. Both only rescaled the
  SatAgg loss.
- **Model selection on the test set.** Some LTN runs selected or monitored the checkpoint with `test_loader`.
  Others saved a best checkpoint but never loaded it. Checkpoint files were shared between scripts. Every
  configuration used a different stopping rule.
- **Models.** `LSTMModel` computed sequence lengths from the first feature, which is not the activity, so on
  sepsis it ran on `InfectionSuspected`. The A variants used a different architecture from the plain models
  (unidirectional vs bidirectional LSTM, different embedding sizes). Every model class reset the global RNG in its
  constructor with its own seed.
- **Semantic loss.** BPI12 used its "-> not accepted" rules as positive triggers. A shape mismatch, (B, 1) * (B,),
  averaged the loss over all pairs of the batch. The sepsis validation loss reused the triggers of the last
  training batch.
- **Scripts that crashed:** `main_bpi12_feat_eng.py` (empty `y_pred_corrected`), `main_traffic_semantic.py`
  (undefined `count_r2`), and the `sepsis_2.csv` scripts (the preprocessing needs `elapsed_time`).

Data fixes that also affect these scripts:

- `data/preprocess_traffic.py` did not write the scaled `paymentAmount` back to the data, so the model saw raw
  payments. The payment rule compared raw payments with the scaled fine amount and practically never fired.
- The train/test split of BPI17, Sepsis and Traffic depended on Python's per-process string hashing: it sampled
  from `list(set(case_ids))`. It was different at every run. It is now sampled from a sorted list.
  BPI12 (integer case ids) was already stable and is unchanged.

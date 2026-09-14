# Compliance-Aware Predictive Process Monitoring: A Neuro-Symbolic Approach

This repository contains the code for the paper:
> **Compliance-Aware Predictive Process Monitoring: A Neuro-Symbolic Approach**

---

## Files

*   **`main_bpi12.py`**: Contains the code to reproduce the experiments with the *BPIC2012* event log.
*   **`main_bpi17.py`**: Contains the code to reproduce the experiments with the *BPIC2017* event log.
*   **`main_sepsis.py`**: Contains the code to reproduce the experiments with the *SEPSIS* event log.
*   **`main_traffic.py`**: Contains the code to reproduce the experiments with the *TRAFFIC FINES* dataset.
*   **`main_feat_eng.py`**: Contains the code to reproduce the experiments with the LSTM-FE and TFR-FE models.
*   **`data/preprocess_bpi12.py`**: Contains the code for preprocessing the *BPIC2012* event log.
*   **`data/preprocess_bpi17.py`**: Contains the code for preprocessing the *BPIC2017* event log.
*   **`data/preprocess_sepsis.py`**: Contains the code for preprocessing the *Sepsis* event log.
*   **`data/preprocess_traffic.py`**: Contains the code for preprocessing the *TRAFFIC FINES* event log.
*   **`model/lstm.py`**: Contains the architecture used for the LSTM backbone.
*   **`model/transformer.py`**: Contains the architecture used for the Transformer backbone.
*   **`data/dataset.py`**: Dataset class.
*   **`create_temporal_features.py`**: Contains the code to create the temporal features also used for logical rules.
*   **`knowledge_base.txt`**: Contains the six rules used for each event log.

---

## Datasets

The event logs used in the study can be downloaded from the following links:

* [BPIC2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)
* [BPIC2017](https://data.4tu.nl/datasets/34c3f44b-3101-4ea9-8281-e38905c68b8d/1)
* [Sepsis](https://data.4tu.nl/datasets/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/1)
* [Traffic fines](https://data.4tu.nl/datasets/806acd1a-2bf2-4e39-be21-69b8cad10909/1)

---

## Declarative constraints to First-Order Logic formulas

The file **`declare_to_fol_templates.pdf`** contains the translation of declarative constraints intoto first-order logic formulas. The resulting FOL formulas can be implemented in the LTN framework to express control-flow constraints in business processes.

The file **`declare_to_fol.py`** implements the declarative constraints described in the previous PDF as predicates that can be used within the LTN framework.

---

## Reproducibility

Execute the script of interest with following flags:
* --backbone: "lstm" or "transformer"
* --seed: random seed used for parameters and splitting
* --num_epochs: number of training epochs of vanilla models
* --num_epochs_nesy: number of training epochs of LTN models
* --hidden_size: hidden_size of LSTM/Transformer backbones
* --num_layers: LSTM/Transformer layers
* --dropout_rate: dropout_rate for LSTM/Transformer backbones

Example for the *SEPSIS* event log with default parameters:

```python main_sepsis.py --model_type="lstm" --hidden_size=128 --num_layers=2 --seed=42```
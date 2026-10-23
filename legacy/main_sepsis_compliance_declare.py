'''
Sepsis compliance experiments with Declare knowledge mined from the event log.

Same experiments as main_sepsis_compliance.py, but the hand-written temporal rules (R4-R6) of the
"C" configurations (BC, AC, ABC) are replaced by Declare constraints discovered automatically:
    1. Declare constraints are mined with pm4py on the training prefixes only (the same inputs the
       models are trained on, so no test case is used and the activity codes are the model's ones)
    2. only high-quality constraints are kept: prefix-safe (a violation on a prefix is permanent),
       confidence >= --declare_min_confidence, same semantics in pm4py and in the LTN predicates
       (|conf - ltn_conf| <= --declare_max_ltn_gap), no redundant / transitively implied constraints
    3. constraints are turned into label rules when the training prefixes violating them (almost)
       all have the same label:  not C(x) -> not P(x)  or  not C(x) -> P(x)
       (precision >= --declare_min_precision, at least --declare_min_violations violating prefixes,
       precision on the validation set >= --declare_min_val_precision), top --declare_top_k kept
    4. the rules are injected as LTN axioms alongside the other logical rules (R1-R3)
Besides the original metrics, the compliance of the predictions with the Declare rules is reported.

The backbone (--model) and the configurations to train and test (--configs) are chosen from the command line:
    vanilla  backbone trained with BCE, no LTN
    ltn      LTN, data axioms only
    B        LTN + knowledge rules R1-R3 as axioms on P
    A        LTN + knowledge rules R1-R3 as input features of P
    AB, BC, AC, ABC  combinations, C = mined Declare label rules

    python main_sepsis_compliance_declare.py                                 # LSTM, all configurations
    python main_sepsis_compliance_declare.py --configs vanilla BC ABC
    python main_sepsis_compliance_declare.py --model transformer --configs ltn BC
'''
import json
from dataclasses import asdict

import pandas as pd
import ltn
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_metrics, compute_metrics_fa
from collections import Counter
from data import preprocess_sepsis
from data.dataset import NeSyDataset, ModelConfig
from declare import DeclareKB, PM4PyDeclareMiner, activity_vocab, activity_slice, prune_constraints
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import warnings
warnings.filterwarnings("ignore")
dataset = "sepsis"
classes = ['No ICU', 'ICU']

MODELS = ["lstm", "transformer"]
CONFIGS = ["vanilla", "ltn", "B", "A", "AB", "BC", "AC", "ABC"]

def get_args():
    parser = argparse.ArgumentParser()

    # model and configurations
    parser.add_argument("--model", type=str, default="lstm", choices=MODELS, help="Backbone of the neural predicate")
    parser.add_argument("--configs", type=str, nargs="+", default=["all"], choices=CONFIGS + ["all"],
                        help="Configurations to train and test (default: all)")
    parser.add_argument("--metrics_output", type=str, default="metrics_declare.txt", help="File the metrics are appended to")

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers of the LSTM / transformer model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate of the LSTM / transformer model")
    parser.add_argument("--model_dim", type=int, default=64, help="Model dimension of the transformer model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads of the transformer model")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=15, help="Number of epochs for training LTN model")
    parser.add_argument("--dataset", type=str, default="sepsis", help="Dataset to use")

    # Declare knowledge
    parser.add_argument("--declare_min_support", type=float, default=0.05, help="Minimum activation support of the mined constraints")
    parser.add_argument("--declare_min_confidence", type=float, default=0.9, help="Minimum confidence of the mined constraints")
    parser.add_argument("--declare_max_ltn_gap", type=float, default=0.01, help="Maximum |pm4py confidence - LTN confidence|")
    parser.add_argument("--declare_min_precision", type=float, default=0.95, help="Minimum precision of a label rule on the training prefixes")
    parser.add_argument("--declare_min_violations", type=int, default=20, help="Minimum number of training prefixes violating a constraint")
    parser.add_argument("--declare_min_val_precision", type=float, default=0.9, help="Minimum precision of a label rule on the validation prefixes")
    parser.add_argument("--declare_top_k", type=int, default=10, help="Maximum number of Declare label rules")
    parser.add_argument("--declare_output", type=str, default="results/declare_sepsis_rules.json", help="Where to save the selected rules")

    args = parser.parse_args()
    # Canonical order, without duplicates
    args.configs = CONFIGS if "all" in args.configs else [c for c in CONFIGS if c in args.configs]
    return args

args = get_args()
print(f"-- Model: {args.model}, configurations: {args.configs}")

dataset = "sepsis"
max_prefix_length = 13

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs = args.num_epochs,
    sequence_length = max_prefix_length,
    dataset = dataset
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("-- Reading dataset")
data = pd.read_csv("data_processed/sepsis_3.csv", dtype={"org:resource": str})

(X_train, y_train, X_val, y_val,  X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_sepsis.preprocess_eventlog(data)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

numerical_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid', "time_since_previous", "elapsed_time"]

### SEPSIS RULES
high_lactic_acid = lambda x: (x[:, 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1)
check_sirs_criteria = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x], device=x.device)
rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)

def add_rule_features(x):
    # Input of the A models: x with the three knowledge rules appended, as in compute_metrics_fa
    r1 = high_lactic_acid(x)
    r2 = check_sirs_criteria(x)
    r3 = torch.logical_and(rule_crp_atb(x).bool(), rule_crp_100(x))
    return torch.cat([x] + [r.float().unsqueeze(1).repeat(1, max_prefix_length) for r in (r1, r2, r3)], dim=1)

### DECLARE KNOWLEDGE
print("-- Mining Declare constraints on the training prefixes")
vocab = activity_vocab(data["concept:name"])
x_train_all = torch.stack(train_dataset.sequences)
x_val_all = torch.stack(val_dataset.sequences)
y_train_all = torch.tensor(y_train)

miner = PM4PyDeclareMiner(min_support=args.declare_min_support, min_confidence=args.declare_min_confidence,
                          prefix_safe_only=True, prune_redundant=False)
mined = miner.fit(activity_slice(x_train_all, feature_names, max_prefix_length), y_train_all, vocab=vocab)
constraints = prune_constraints(mined, max_ltn_gap=args.declare_max_ltn_gap, redundant=True, transitive=True,
                                sort_by=["confidence", "activation_support"])
kb = DeclareKB(constraints, feature_names, max_prefix_length, vocab=vocab)
print(f"Mined {len(mined)} prefix-safe constraints, {len(kb)} after pruning")

candidate_rules = kb.label_rules(x_train_all, y_train_all, min_precision=args.declare_min_precision,
                                 min_violations=args.declare_min_violations)
val_stats = kb.rule_stats(candidate_rules, x_val_all, val_dataset.labels)
declare_rules = [r for r, (val_precision, _) in zip(candidate_rules, val_stats)
                 if val_precision is None or val_precision >= args.declare_min_val_precision][:args.declare_top_k]
val_stats = dict(zip([id(r) for r in candidate_rules], val_stats))
print(f"Declare label rules ({len(declare_rules)} of {len(candidate_rules)} candidates):")
for r in declare_rules:
    val_precision, val_violations = val_stats[id(r)]
    print(f"   {r}  [val precision={'n/a' if val_precision is None else f'{val_precision:.3f}'} val violations={val_violations}]")

with open(args.declare_output, "w") as f:
    json.dump([{"rule": str(r), "mode": r.mode, "precision": r.precision, "violations": r.violations,
                "val_precision": val_stats[id(r)][0], "val_violations": val_stats[id(r)][1],
                "constraint": asdict(r.constraint)} for r in declare_rules], f, indent=2)

def declare_compliance(model, loader, rule_features=False):
    # Fraction of test prefixes violating a Declare label rule (and whose true label agrees with the
    # rule) for which the prediction agrees with the rule
    agree, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            predictions = (model(add_rule_features(x) if rule_features else x).squeeze(-1) > 0.5).long()
            a, t = kb.label_compliance(x, predictions, declare_rules, y=y.to(device))
            agree += a
            total += t
    return agree / total if total else float("nan")

### MODELS

def make_model(rule_features=False):
    '''Backbone selected with --model; rule_features: the input contains the three knowledge rules (A configurations).'''
    if args.model == "lstm":
        model_cls = LSTMModelA if rule_features else LSTMModel
        return model_cls(vocab_sizes, config, 1, feature_names, numerical_features)
    # Imported here so that the LSTM experiments do not depend on the transformers package
    from model.transformer import EventTransformer, EventTransformerA
    model_cls = EventTransformerA if rule_features else EventTransformer
    return model_cls(vocab_sizes, config, feature_names, numerical_features, model_dim=args.model_dim, num_classes=1,
                     max_len=max_prefix_length, num_layers=args.num_layers, num_heads=args.num_heads,
                     dropout=args.dropout_rate)

def print_metrics(title, metrics):
    print(title)
    for name, value in zip(["Accuracy", "F1 Score", "Precision", "Recall", "Compliance", "Declare compliance"], metrics):
        print(f"{name}:", value)

### VANILLA (no LTN)

def run_vanilla():
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()

    model.train()
    training_losses = []
    validation_losses = []
    for epoch in range(config.num_epochs):
        train_losses = []
        for enum, (x, y) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.squeeze(1).cpu(), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
        training_losses.append(statistics.mean(train_losses))
        model.eval()
        val_losses = []
        for enum, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                x = x.to(device)
                output = model(x)
                loss = criterion(output.squeeze(1).cpu(), y)
                val_losses.append(loss.item())
        print(f"Validation Loss: {statistics.mean(val_losses)}")
        validation_losses.append(statistics.mean(val_losses))
        if epoch >= 5:
            if validation_losses[-1] > validation_losses[-2]:
                print("Validation loss increased, stopping training")
                break
        model.train()

    model.eval()
    y_pred = []
    y_true = []
    compliance = 0
    num_constraints = 0
    for enum, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            high_lactic_acid_res = high_lactic_acid(x).detach().cpu().numpy()
            rule_crp_atb_res = rule_crp_atb(x).detach().cpu().numpy()
            rule_crp_100_res = rule_crp_100(x).detach().cpu().numpy()
            outputs = model(x).detach().cpu().numpy()
            predictions = np.where(outputs > 0.5, 1., 0.).flatten()
            for i in range(len(y)):
                y_pred.append(predictions[i])
                y_true.append(y[i].cpu())
                if high_lactic_acid_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        compliance += 1
                if rule_crp_atb_res[i] == 1 and rule_crp_100_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        compliance += 1

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix_{args.model}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return [accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro'),
            precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'),
            compliance / num_constraints, declare_compliance(model, test_loader)]

### LTN

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
SatAgg = ltn.fuzzy_ops.SatAgg()

HighLacticAcid = ltn.Predicate(func=lambda x: (x[:, 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1))
PresentCritCriteria = ltn.Predicate(func=lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1)))
CheckPresenceCRPAtb = ltn.Predicate(func= lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device))
CheckCRP100 = ltn.Predicate(func = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1))

sirs_condition = lambda x: (x.value[:, :13].eq(1).any(dim=1)) & (x.value[:, 39:52].eq(1).any(dim=1)) & (x.value[:, 65:78].eq(1).any(dim=1))

def check_time_constraint(x):
    batch_results = []
    for i in range(x.value.shape[0]):
        row = x.value[i]
        # Find positions where activity 5 and 6 occur
        act5_positions = (row[104:117] == 5).nonzero(as_tuple=True)[0]
        act6_positions = (row[104:117] == 6).nonzero(as_tuple=True)[0]

        # Check if both activities exist
        if len(act5_positions) > 0 and len(act6_positions) > 0:
            # Get elapsed times for these positions
            times_act5 = row[-13:][act5_positions]
            times_act6 = row[-13:][act6_positions]

            # Check if any pair has time difference <= 120
            satisfies = False
            for t5 in times_act5:
                for t6 in times_act6:
                    if abs(t5 - t6) <= scalers["elapsed_time"].transform([[120]])[0][0]:
                        satisfies = True
                        break
                if satisfies:
                    break
            batch_results.append(satisfies)
        else:
            batch_results.append(False)

    return torch.tensor(batch_results, dtype=torch.bool, device=x.value.device)

def run_ltn(name):
    '''
    Train and test one LTN configuration:
        A: P takes the knowledge rules R1-R3 as input features, and their satisfaction is an axiom
        B: R1-R3 are axioms on P
        C: the mined Declare label rules are axioms on P
    '''
    use_A, use_B, use_C = "A" in name, "B" in name, "C" in name
    model = make_model(rule_features=use_A)
    P = ltn.Predicate(model).to(device)
    optimizer = torch.optim.Adam(P.parameters(), lr=config.learning_rate)

    for epoch in range(args.num_epochs_nesy):
        train_loss = 0.0
        for enum, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_in = add_rule_features(x) if use_A else x  # input of P
            x_P = ltn.Variable("x_P", x_in[y==1])
            x_not_P = ltn.Variable("x_not_P", x_in[y==0])
            x_All = ltn.Variable("x_All", x_in)
            x_All_A = ltn.Variable("x_All_A", x)
            formulas = []
            if x_P.value.numel()>0:
                formulas.append(Forall(x_P, P(x_P)))
            if x_not_P.value.numel()>0:
                formulas.append(Forall(x_not_P, Not(P(x_not_P))))
            if use_B:
                # SEPSIS KG
                formulas.extend([
                    Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn=sirs_condition),
                    Forall(x_All, Implies(PresentCritCriteria(x_All), P(x_All))),
                    Forall(x_All, Implies(And(CheckPresenceCRPAtb(x_All), CheckCRP100(x_All)), P(x_All))),
                ])
                if name == "B":
                    # As in main_sepsis_compliance.py, the time constraint is only used by the B configuration
                    formulas.append(Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=check_time_constraint))
            if use_A:
                formulas.extend([
                    Forall(x_All_A, HighLacticAcid(x_All_A)),
                    Forall(x_All_A, PresentCritCriteria(x_All_A)),
                    Forall(x_All_A, And(CheckPresenceCRPAtb(x_All_A), CheckCRP100(x_All_A))),
                ])
            if use_C:
                # Declare knowledge: not C(x) -> (not) P(x)
                formulas.extend(kb.label_axioms(x_All, P, declare_rules))
            sat_agg = SatAgg(*formulas)
            loss = 1 - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del x_P, x_not_P, sat_agg
        train_loss = train_loss / len(train_loader)
        print(" epoch %d | loss %.4f"
                    %(epoch, train_loss))

    model.eval()
    # Same evaluation functions as main_sepsis_compliance.py
    evaluate = compute_metrics if name in ("ltn", "B") else compute_metrics_fa
    metrics = list(evaluate(test_loader, model, device, "ltn" if name == "ltn" else "ltn_w_k", scalers, dataset))
    return metrics + [declare_compliance(model, test_loader, rule_features=use_A)]

### EXPERIMENTS

results = {}
for name in args.configs:
    print(f"-- Training {args.model} {name}")
    results[name] = run_vanilla() if name == "vanilla" else run_ltn(name)
    print_metrics(f"Metrics {args.model} {name}", results[name])

with open(args.metrics_output, "a") as f:
    f.write("Accuracy, F1, Precision, Recall, Compliance, Declare compliance\n")
    f.write(f"Model: {args.model}\n")
    f.write("Declare rules: " + "; ".join(str(r) for r in declare_rules) + "\n")
    for name, metrics in results.items():
        label = args.model.upper() if name == "vanilla" else ("LTN" if name == "ltn" else f"LTN_{name}")
        f.write(f"{label}: \n")
        f.write(str(metrics))
        f.write("\n")

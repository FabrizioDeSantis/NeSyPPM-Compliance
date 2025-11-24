import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA
from model.transformer import EventTransformer, EventTransformerA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import statistics
from metrics import compute_metrics, compute_metrics_fa
from collections import defaultdict, Counter
from data import preprocess_bpi12
from data.dataset import NeSyDataset, ModelConfig

import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "bpi12"
classes = ["Not accepted", "Accepted"]

metrics_lstm = []
metrics_ltn = []
metrics_ltn_A = []
metrics_ltn_B = []
metrics_ltn_AB = []
metrics_ltn_BC = []
metrics_ltn_AC = []
metrics_ltn_ABC = []

def get_args():
    parser = argparse.ArgumentParser()

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=5, help="Number of epochs for training LTN model")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--backbone", type=str, default="lstm", help="Model backbone: lstm or transformer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

args = get_args()

dataset = "bpi12"
max_prefix_length = 40

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
data = pd.read_csv("data_processed/"+dataset+".csv", dtype={"org:resource": str})

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(data, args.seed)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Validation set")
counts = Counter(y_val)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

numerical_features = ["case:AMOUNT_REQ"]

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

lstm.train()
training_losses = []
validation_losses = []
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output.squeeze(1), y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
    training_losses.append(statistics.mean(train_losses))
    lstm.eval()
    val_losses = []
    for enum, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            output = lstm(x)
            loss = criterion(output.squeeze(1), y)
            val_losses.append(loss.item())
    print(f"Validation Loss: {statistics.mean(val_losses)}")
    validation_losses.append(statistics.mean(val_losses))
    if epoch >= 5:
        if validation_losses[-1] > validation_losses[-2]:
            print("Validation loss increased, stopping training")
            break
    lstm.train()

lstm.eval()
y_pred = []
y_true = []
# BPI12 RULES
rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)
compliance_lstm = 0
num_constraints = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        # Apply the rule to the input data
        rule_amount_1_res = rule_amount_1(x).detach().cpu().numpy()
        rule_amount_2_res = rule_amount_2(x).detach().cpu().numpy()
        rule_amount_3_res = rule_amount_3(x).detach().cpu().numpy()
        rule_resource_1_res = rule_resource_1(x).detach().cpu().numpy()
        rule_resource_2_res = rule_resource_2(x).detach().cpu().numpy()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if rule_amount_1_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_amount_2_res[i] == 1 and rule_amount_3_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_resource_1_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_resource_2_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1

print("Metrics LSTM")
accuracy = accuracy_score(y_true, y_pred)
metrics_lstm.append(accuracy)
print("Accuracy:", accuracy)
f1 = f1_score(y_true, y_pred, average='macro')
metrics_lstm.append(f1)
print("F1 Score:", f1)
precision = precision_score(y_true, y_pred, average='macro')
metrics_lstm.append(precision)
print("Precision:", precision)
recall = recall_score(y_true, y_pred, average='macro')
metrics_lstm.append(recall)
print("Recall:", recall)
print("Compliance:", compliance_lstm / num_constraints)
metrics_lstm.append(compliance_lstm / num_constraints)

# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def compute_satisfaction_level(loader, MainP):
    mean_sat = 0
    for enum, (x, y) in enumerate(loader):
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, MainP(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(MainP(x_not_P)))
            ])
        mean_sat += SatAgg(
            *formulas
        ).detach().cpu()
        del x_P, x_not_P
    mean_sat /= len(loader)
    return mean_sat

for epoch in range(args.num_epochs_nesy):
    lstm.train()
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    with torch.no_grad():
        lstm.eval()
        _, f1_val, _, _, _ = compute_metrics(test_loader, lstm, device, "ltn", scalers, dataset)
    print(" epoch %d | loss %.4f | f1 val %.4f"
                %(epoch, train_loss, f1_val))

lstm.eval()
print("Metrics LTN w/o knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn.append(f1score)
print("Precision:", precision)
metrics_ltn.append(precision)
print("Recall:", recall)
metrics_ltn.append(recall)
print("Compliance:", compliance)
metrics_ltn.append(compliance)

# LTN_B

# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

f1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
f2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
f3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
IsAmountReqLessThan10k = ltn.Predicate(func = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1))
IsAmountReqGreaterThan50k = ltn.Predicate(func = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1))
IsAmountLessThan60k = ltn.Predicate(func = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))
Res11169ExecutedActivity = ltn.Predicate(func = lambda x: (x[:, :240] == 48).any(dim=1))
Res10910ExecutedActivity = ltn.Predicate(func = lambda x: (x[:, :240] == 21).any(dim=1))
f_resources_11169 = lambda x: (x[:, :240] == 48).any(dim=1)
f_resources_10910 = lambda x: (x[:, :240] == 21).any(dim=1)

max_f1_val = 0
for epoch in range(args.num_epochs_nesy):
    lstm.train()
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            # BPI 12 KG
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            # Forall(x_All, Implies(IsAmountReqLessThan10k(x_All), Not(P(x_All)))),
            # Forall(x_All, Implies(And(IsAmountReqGreaterThan50k(x_All), IsAmountLessThan60k(x_All)), Not(P(x_All)))),
            # Forall(x_All, Implies(Res11169ExecutedActivity(x_All), Not(P(x_All)))),
            # Forall(x_All, Implies(Res10910ExecutedActivity(x_All), Not(P(x_All)))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    with torch.no_grad():
        lstm.eval()
        _, f1_val, _, _, _ = compute_metrics(val_loader, lstm, device, "ltn", scalers, dataset)
        if f1_val > max_f1_val:
            max_f1_val = f1_val
            torch.save(lstm.state_dict(), "best_model_ltn.pth")
    print(" epoch %d | loss %.4f | f1 val %.4f"
                %(epoch, train_loss, f1_val))
    
lstm.load_state_dict(torch.load("best_model_ltn.pth"))
lstm.eval()
print("Metrics LTN w knowledge (B)")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_B.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_B.append(f1score)
print("Precision:", precision)
metrics_ltn_B.append(precision)
print("Recall:", recall)
metrics_ltn_B.append(recall)
print("Compliance:", compliance)
metrics_ltn_B.append(compliance)

# LTN_A

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        rule_1_res = f1(x).detach()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_P = ltn.Variable("x_P", x_concat[y==1])
        x_not_P = ltn.Variable("x_not_P", x_concat[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, IsAmountReqLessThan10k(x_All)),
            Forall(x_All, And(IsAmountReqGreaterThan50k(x_All), IsAmountLessThan60k(x_All))),
            Forall(x_All, Or(Res11169ExecutedActivity(x_All), Res10910ExecutedActivity(x_All))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    with torch.no_grad():
        lstm.eval()
        _, f1_val, _, _, _ = compute_metrics_fa(test_loader, lstm, device, "ltn", scalers, dataset)
    print(" epoch %d | loss %.4f | f1 val %.4f"
                %(epoch, train_loss, f1_val))
    lstm.train()

lstm.eval()
print("Metrics LTN w knowledge (A)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_A.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_A.append(f1score)
print("Precision:", precision)
metrics_ltn_A.append(precision)
print("Recall:", recall)
metrics_ltn_A.append(recall)
print("Compliance:", compliance)
metrics_ltn_A.append(compliance)

# LTN_AB

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        rule_1_res = f1(x).detach()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_P = ltn.Variable("x_P", x_concat[y==1])
        x_not_P = ltn.Variable("x_not_P", x_concat[y==0])
        x_All = ltn.Variable("x_All", x_concat)
        x_All_A = ltn.Variable("x_All_A", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P))),
            ])
        formulas.extend([
            # BPI 12 KG
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            Forall(x_All_A, IsAmountReqLessThan10k(x_All_A)),
            Forall(x_All_A, And(IsAmountReqGreaterThan50k(x_All_A), IsAmountLessThan60k(x_All_A))),
            Forall(x_All_A, Or(Res11169ExecutedActivity(x_All_A), Res10910ExecutedActivity(x_All_A))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w knowledge (AB)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_AB.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_AB.append(f1score)
print("Precision:", precision)
metrics_ltn_AB.append(precision)
print("Recall:", recall)
metrics_ltn_AB.append(recall)
print("Compliance:", compliance)
metrics_ltn_AB.append(compliance)

# LTN_BC

# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
HasAct = ltn.Predicate(func = lambda x, act: torch.tensor(x[:, 104:117] == act[0].item()).any(dim=1))
IsNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i < j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))
IsImmediateNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i + 1 == j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))
P = ltn.Predicate(lstm).to(device)
SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)
W_Completeren_aanraag_COMPLETE = ltn.Constant(torch.tensor([22]))
W_Valideren_aanvraag_COMPLETE = ltn.Constant(torch.tensor([31]))
O_SENT_BACK_COMPLETE = ltn.Constant(torch.tensor([15]))
O_CANCELLED_COMPLETE = ltn.Constant(torch.tensor([11]))
A_ACCEPTED_COMPLETE = ltn.Constant(torch.tensor([1]))

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            Forall(x_All, And(HasAct(x_All, W_Completeren_aanraag_COMPLETE), And(HasAct(x_All, A_ACCEPTED_COMPLETE), IsNext(x_All, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))),
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, W_Completeren_aanraag_COMPLETE), And(HasAct(x, A_ACCEPTED_COMPLETE), IsNext(x, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))).value > 0),
            Forall(x_All, And(HasAct(x_All, O_SENT_BACK_COMPLETE), And(HasAct(x_All, W_Valideren_aanvraag_COMPLETE), IsNext(x_All, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))),
                   cond_vars=[x_All],
                     cond_fn = lambda x: And(HasAct(x, O_SENT_BACK_COMPLETE), And(HasAct(x, W_Valideren_aanvraag_COMPLETE), IsNext(x, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))).value > 0),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w knowledge and parallel constraints")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_BC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_BC.append(f1score)
print("Precision:", precision)
metrics_ltn_BC.append(precision)
print("Recall:", recall)
metrics_ltn_BC.append(recall)
print("Compliance:", compliance)
metrics_ltn_BC.append(compliance)

# LTN_AC

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
# for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_All = ltn.Variable("x_All", x)
        rule_1_res = f1(x).detach().cpu()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_P = ltn.Variable("x_P", x_concat[y==1])
        x_not_P = ltn.Variable("x_not_P", x_concat[y==0])
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, IsAmountReqLessThan10k(x_All)),
            Forall(x_All, And(IsAmountReqGreaterThan50k(x_All), IsAmountLessThan60k(x_All))),
            Forall(x_All, Or(Res11169ExecutedActivity(x_All), Res10910ExecutedActivity(x_All))),
            Forall(x_All, And(HasAct(x_All, W_Completeren_aanraag_COMPLETE), And(HasAct(x_All, A_ACCEPTED_COMPLETE), IsNext(x_All, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))),
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, W_Completeren_aanraag_COMPLETE), And(HasAct(x, A_ACCEPTED_COMPLETE), IsNext(x, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))).value > 0),
            Forall(x_All, And(HasAct(x_All, O_SENT_BACK_COMPLETE), And(HasAct(x_All, W_Valideren_aanvraag_COMPLETE), IsNext(x_All, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))),
                   cond_vars=[x_All],
                     cond_fn = lambda x: And(HasAct(x, O_SENT_BACK_COMPLETE), And(HasAct(x, W_Valideren_aanvraag_COMPLETE), IsNext(x, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))).value > 0),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

lstm.eval()
print("Metrics LTN w knowledge (AC)")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_AC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_AC.append(f1score)
print("Precision:", precision)
metrics_ltn_AC.append(precision)
print("Recall:", recall)
metrics_ltn_AC.append(recall)
print("Compliance:", compliance)
metrics_ltn_AC.append(compliance)

# LTN_ABC

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)
max_f1_val = 0.0
for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        rule_1_res = f1(x).detach()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_P = ltn.Variable("x_P", x_concat[y==1])
        x_not_P = ltn.Variable("x_not_P", x_concat[y==0])
        x_All = ltn.Variable("x_All", x_concat)
        x_All_A = ltn.Variable("x_All_A", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1) & (x.value[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 48).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :240] == 21).any(dim=1)),
            Forall(x_All_A, IsAmountReqLessThan10k(x_All_A)),
            Forall(x_All_A, And(IsAmountReqGreaterThan50k(x_All_A), IsAmountLessThan60k(x_All_A))),
            Forall(x_All_A, Or(Res11169ExecutedActivity(x_All_A), Res10910ExecutedActivity(x_All_A))),
            Forall(x_All_A, And(HasAct(x_All_A, W_Completeren_aanraag_COMPLETE), And(HasAct(x_All_A, A_ACCEPTED_COMPLETE), IsNext(x_All_A, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))),
                   cond_vars=[x_All_A],
                   cond_fn = lambda x: And(HasAct(x, W_Completeren_aanraag_COMPLETE), And(HasAct(x, A_ACCEPTED_COMPLETE), IsNext(x, W_Completeren_aanraag_COMPLETE, A_ACCEPTED_COMPLETE))).value > 0),
            Forall(x_All_A, And(HasAct(x_All_A, O_SENT_BACK_COMPLETE), And(HasAct(x_All_A, W_Valideren_aanvraag_COMPLETE), IsNext(x_All_A, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))),
                   cond_vars=[x_All_A],
                     cond_fn = lambda x: And(HasAct(x, O_SENT_BACK_COMPLETE), And(HasAct(x, W_Valideren_aanvraag_COMPLETE), IsNext(x, O_SENT_BACK_COMPLETE, W_Valideren_aanvraag_COMPLETE))).value > 0),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    with torch.no_grad():
        lstm.eval()
        _, f1_val, _, _, _ = compute_metrics_fa(test_loader, lstm, device, "ltn", scalers, dataset)
        if f1_val > max_f1_val:
            max_f1_val = f1_val
            torch.save(lstm.state_dict(), "best_ltn_model.pth")
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))
    lstm.train()

lstm.load_state_dict(torch.load("best_ltn_model.pth"))

lstm.eval()
print("Metrics LTN w knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics_fa(test_loader, lstm, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_ABC.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_ABC.append(f1score)
print("Precision:", precision)
metrics_ltn_ABC.append(precision)
print("Recall:", recall)
metrics_ltn_ABC.append(recall)
print("Compliance:", compliance)
metrics_ltn_ABC.append(compliance)

with open("metrics.txt", "a") as f:
    f.write("Accuracy, F1, Precision, Recall, Compliance\n")
    f.write("LSTM: \n")
    f.write(str(metrics_lstm))
    f.write("\n")
    f.write("LTN: \n")
    f.write(str(metrics_ltn))
    f.write("\n")
    f.write("LNT_B: \n")
    f.write(str(metrics_ltn_B))
    f.write("\n")
    f.write("LTN_A: \n")
    f.write(str(metrics_ltn_A))
    f.write("\n")
    f.write("LTN_AB: \n")
    f.write(str(metrics_ltn_AB))
    f.write("\n")
    f.write("LTN_BC: \n")
    f.write(str(metrics_ltn_BC))
    f.write("\n")
    f.write("LTN_AC: \n")
    f.write(str(metrics_ltn_AC))
    f.write("\n")
    f.write("LTN_ABC: \n")
    f.write(str(metrics_ltn_ABC))
    f.write("\n")
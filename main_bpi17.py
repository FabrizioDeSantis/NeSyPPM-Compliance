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
from data import preprocess_bpi17
from data.dataset import NeSyDataset, ModelConfig

import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "bpi17"
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

dataset = "bpi17"
max_prefix_length = 20

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

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi17.preprocess_eventlog(data, args.seed)

numerical_features = ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount", "elapsed_time", "time_since_previous"]

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

if args.backbone == "lstm":
    lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
else:
    lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
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
compliance_lstm = 0
num_constraints = 0
rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        # Apply the rule to the input data
        rule_1_result = rule_1(x).detach().cpu().numpy()
        rule_2_result = rule_2(x).detach().cpu().numpy()
        rule_3_result = rule_3(x).detach().cpu().numpy()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if rule_1_result[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
                    compliance_lstm += 1
            if rule_2_result[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_3_result[i] == 1 and y[i] == 0:
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
print(num_constraints)
print("Compliance:", compliance_lstm / num_constraints)
metrics_lstm.append(compliance_lstm / num_constraints)

if args.backbone == "lstm":
    lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
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

IsCreditScoreGreaterThan0 = ltn.Predicate(func = lambda x: (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
IsRequestedAmountGreaterThan20k = ltn.Predicate(func = lambda x: x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0])
NoOfferWithCreditScoreGreaterThan0 = ltn.Predicate(func = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1))
LoanGoalExistingLoanTakeover = ltn.Predicate(func = lambda x: (x[:, 20:40] == 6).any(dim=1))

HasAct = ltn.Predicate(func = lambda x, act: torch.tensor(x[:, 104:117] == act[0].item()).any(dim=1))
IsNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i < j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))
IsImmediateNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i + 1 == j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel() > 0:
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
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

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

if args.backbone == "lstm":
    lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
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

max_f1_val = 0
for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel() > 0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn=lambda x: ((x.value[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, :20] == 11).any(dim=1) & (x.value[:, 40:60] != 0).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 40:60] == 0).all(dim=1) & (x.value[:, :20] == 11).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, 20:40] == 6).any(dim=1)),
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
        _, f1, _, _, _ = compute_metrics(val_loader, lstm, device, "ltn", scalers, dataset)
        if f1 > max_f1_val:
            max_f1_val = f1
            torch.save(lstm.state_dict(), "best_model_lstm.pth")
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))
    lstm.train()

lstm.load_state_dict(torch.load("best_model_lstm.pth"))
lstm.eval()
print("Metrics LTN w knowledge")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, lstm, device, "ltn", scalers, dataset)
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

if args.backbone == "lstm":
    lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        x_All = ltn.Variable("x_All", x)
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
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
            Forall(x_All, And(IsCreditScoreGreaterThan0(x_All), Not(IsRequestedAmountGreaterThan20k(x_All)))),
            Forall(x_All, NoOfferWithCreditScoreGreaterThan0(x_All)),
            Forall(x_All, And(IsRequestedAmountGreaterThan20k(x_All), LoanGoalExistingLoanTakeover(x_All))),
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

if args.backbone == "lstm":
    lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        x_All_A = ltn.Variable("x_All_A", x)
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
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
                Forall(x_not_P, Not(P(x_not_P))),
            ])
        formulas.extend([
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn=lambda x: ((x.value[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, :20] == 11).any(dim=1) & (x.value[:, 40:60] != 0).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 40:60] == 0).all(dim=1) & (x.value[:, :20] == 11).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, 20:40] == 6).any(dim=1)),
            Forall(x_All_A, And(IsCreditScoreGreaterThan0(x_All_A), Not(IsRequestedAmountGreaterThan20k(x_All_A)))),
            Forall(x_All_A, NoOfferWithCreditScoreGreaterThan0(x_All_A)),
            Forall(x_All_A, And(IsRequestedAmountGreaterThan20k(x_All_A), LoanGoalExistingLoanTakeover(x_All_A))),
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

if args.backbone == "lstm":
    lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)
SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

HasAct = ltn.Predicate(func = lambda x, act: torch.tensor(x[:, 104:117] == act[0].item()).any(dim=1))
IsNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i < j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))
IsImmediateNext = ltn.Predicate(func = lambda x, act1, act2: torch.tensor([int(any(i + 1 == j for i in (row[104:117] == act1[0].item()).nonzero(as_tuple=True)[0] for j in (row[104:117] == act2[0].item()).nonzero(as_tuple=True)[0])) for row in x]).to(device))
A_Submitted = ltn.Constant(torch.tensor([8]))
A_Accepted = ltn.Constant(torch.tensor([1]))
A_Complete = ltn.Constant(torch.tensor([3]))
O_Create_Offer = ltn.Constant(torch.tensor([11]))
W_validate_application = ltn.Constant(torch.tensor([23]))

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn=lambda x: ((x.value[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, :20] == 11).any(dim=1) & (x.value[:, 40:60] != 0).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 40:60] == 0).all(dim=1) & (x.value[:, :20] == 11).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, 20:40] == 6).any(dim=1)),
        ])
        formulas.extend([
            Forall(x_All, And(HasAct(x_All, A_Submitted), And(HasAct(x_All, A_Accepted), IsNext(x_All, A_Submitted, A_Accepted))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Submitted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Submitted, A_Accepted))).value > 0),
            Forall(x_All, And(HasAct(x_All, A_Accepted), And(HasAct(x_All, O_Create_Offer), IsNext(x_All, A_Accepted, O_Create_Offer))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Accepted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Accepted, O_Create_Offer))).value > 0),
            Forall(x_All, And(HasAct(x_All, A_Complete), And(HasAct(x_All, W_validate_application), IsNext(x_All, A_Complete, W_validate_application))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Complete), And(HasAct(x, W_validate_application), IsImmediateNext(x, A_Complete, W_validate_application))).value > 0),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f "
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

if args.backbone == "lstm":
    lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        x_Next = ltn.Variable("x_All", x)
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, 20)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, 20)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, 20)], dim=1)
        optimizer.zero_grad()
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
            Forall(x_All, And(IsCreditScoreGreaterThan0(x_All), Not(IsRequestedAmountGreaterThan20k(x_All)))),
            Forall(x_All, NoOfferWithCreditScoreGreaterThan0(x_All)),
            Forall(x_All, And(IsRequestedAmountGreaterThan20k(x_All), LoanGoalExistingLoanTakeover(x_All))),
            Forall(x_All, And(HasAct(x_All, A_Submitted), And(HasAct(x_All, A_Accepted), IsNext(x_All, A_Submitted, A_Accepted))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Submitted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Submitted, A_Accepted))).value > 0),
            Forall(x_All, And(HasAct(x_All, A_Accepted), And(HasAct(x_All, O_Create_Offer), IsNext(x_All, A_Accepted, O_Create_Offer))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Accepted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Accepted, O_Create_Offer))).value > 0),
            Forall(x_All, And(HasAct(x_All, A_Complete), And(HasAct(x_All, W_validate_application), IsNext(x_All, A_Complete, W_validate_application))), 
                   cond_vars=[x_All],
                   cond_fn = lambda x: And(HasAct(x, A_Complete), And(HasAct(x, W_validate_application), IsImmediateNext(x, A_Complete, W_validate_application))).value > 0),
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

if args.backbone == "lstm":
    lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features)
else:
    lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
P = ltn.Predicate(lstm).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
#for epoch in range(1):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        x_concat = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, 20)], dim=1)
        x_concat = torch.cat([x_concat, rule_2_res.unsqueeze(1).repeat(1, 20)], dim=1)
        x_concat = torch.cat([x_concat, rule_3_res.unsqueeze(1).repeat(1, 20)], dim=1)
        optimizer.zero_grad()
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
            Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn=lambda x: ((x.value[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, :20] == 11).any(dim=1) & (x.value[:, 40:60] != 0).any(dim=1))),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 40:60] == 0).all(dim=1) & (x.value[:, :20] == 11).any(dim=1)),
            Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=lambda x: (x.value[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x.value[:, 20:40] == 6).any(dim=1)),
            Forall(x_All_A, And(IsCreditScoreGreaterThan0(x_All_A), Not(IsRequestedAmountGreaterThan20k(x_All_A)))),
            Forall(x_All_A, NoOfferWithCreditScoreGreaterThan0(x_All_A)),
            Forall(x_All_A, And(IsRequestedAmountGreaterThan20k(x_All_A), LoanGoalExistingLoanTakeover(x_All_A))),
            Forall(x_All_A, And(HasAct(x_All_A, A_Submitted), And(HasAct(x_All_A, A_Accepted), IsNext(x_All_A, A_Submitted, A_Accepted))), 
                   cond_vars=[x_All_A],
                   cond_fn = lambda x: And(HasAct(x, A_Submitted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Submitted, A_Accepted))).value > 0),
            Forall(x_All_A, And(HasAct(x_All_A, A_Accepted), And(HasAct(x_All_A, O_Create_Offer), IsNext(x_All_A, A_Accepted, O_Create_Offer))), 
                   cond_vars=[x_All_A],
                   cond_fn = lambda x: And(HasAct(x, A_Accepted), And(HasAct(x, A_Accepted), IsImmediateNext(x, A_Accepted, O_Create_Offer))).value > 0),
            Forall(x_All_A, And(HasAct(x_All_A, A_Complete), And(HasAct(x_All_A, W_validate_application), IsNext(x_All_A, A_Complete, W_validate_application))), 
                   cond_vars=[x_All_A],
                   cond_fn = lambda x: And(HasAct(x, A_Complete), And(HasAct(x, W_validate_application), IsImmediateNext(x, A_Complete, W_validate_application))).value > 0),
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
        _, f1, _, _, _ = compute_metrics_fa(val_loader, lstm, device, "ltn", scalers, dataset)
        if f1 > max_f1_val:
            max_f1_val = f1
            torch.save(lstm.state_dict(), "best_model_lstm.pth")
    print(" epoch %d | loss %.4f "
                %(epoch, train_loss))
    lstm.train()

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
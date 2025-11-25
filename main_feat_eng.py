import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_metrics, compute_metrics_fa
from collections import defaultdict, Counter
from data import preprocess_sepsis
from data.dataset import NeSyDataset, ModelConfig
from model.semantic_loss import semantic_loss, semantic_loss_pos
from model.transformer import EventTransformer, EventTransformerA
import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

def get_args():
    parser = argparse.ArgumentParser()

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=15, help="Number of epochs for training LTN model")
    parser.add_argument("--dataset", type=str, default="sepsis", help="Dataset to use")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--model", type=str, default="lstm", help="Model to use: lstm or transformer")

    return parser.parse_args()

args = get_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("-- Reading dataset")
data = pd.read_csv("data_processed/"+args.dataset+".csv", dtype={"org:resource": str})
(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_sepsis.preprocess_eventlog(data)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

if args.dataset == "sepsis":
    numerical_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid', "elapsed_time", "time_since_previous"]
    max_prefix_length = 13
elif args.dataset == "bpi12":
    numerical_features = ["case:AMOUNT_REQ", "elapsed_time", "time_since_previous"]
    max_prefix_length = 40
elif args.dataset == "bpi17":
    numerical_features = ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount", "elapsed_time", "time_since_previous"]
    max_prefix_length = 20
elif args.dataset == "traffic":
    numerical_features = ["expense", "amount", "paymentAmount", "elapsed_time", "time_since_previous"]
    max_prefix_length = 10

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs = args.num_epochs,
    sequence_length = max_prefix_length,
    dataset = args.dataset
)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if args.model == "lstm":
    lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
else:
    lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)

optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

if args.dataset == "sepsis":
    rule_1 = lambda x: (x[..., 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1)
    rule_2 = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
    rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device)
    rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)
elif args.dataset == "bpi12":
    f1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
    f2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
    f3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
    f_resources_11169 = lambda x: (x[:, :240] == 48).any(dim=1)
    f_resources_10910 = lambda x: (x[:, :240] == 21).any(dim=1)
elif args.dataset == "bpi17":
    rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
    rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
    rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)
elif args.dataset == "traffic":
    rule_penalty = lambda x: (x[:, :10] == 1).any(dim=1)
    rule_payment = lambda x: (x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])
    rule_amount = lambda x: (x[:, 90] > scalers["amount"].transform([[400]])[0][0])

lstm.train()
training_losses = []
validation_losses = []
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        if args.dataset == "sepsis":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = torch.logical_and(rule_crp_atb(x).detach(), rule_crp_100(x).detach())
        elif args.dataset == "bpi12":
            rule_1_res = f1(x).detach()
            f2_res = f2(x).detach()
            f3_res = f3(x).detach()
            f_resources_11169_res = f_resources_11169(x).detach()
            f_resources_10910_res = f_resources_10910(x).detach()
            rule_2_res = torch.logical_and(f2_res, f3_res).detach()
            rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        elif args.dataset == "bpi17":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = rule_3(x).detach()
        elif args.dataset == "traffic":
            rule_1_res = rule_penalty(x).detach()
            rule_2_res = rule_payment(x).detach()
            rule_3_res = rule_amount(x).detach()
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
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
            if args.dataset == "sepsis":
                rule_1_res = rule_1(x).detach()
                rule_2_res = rule_2(x).detach()
                rule_3_res = torch.logical_and(rule_crp_atb(x).detach(), rule_crp_100(x).detach())
            elif args.dataset == "bpi12":
                rule_1_res = f1(x).detach()
                f2_res = f2(x).detach()
                f3_res = f3(x).detach()
                f_resources_11169_res = f_resources_11169(x).detach()
                f_resources_10910_res = f_resources_10910(x).detach()
                rule_2_res = torch.logical_and(f2_res, f3_res).detach()
                rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
            elif args.dataset == "bpi17":
                rule_1_res = rule_1(x).detach()
                rule_2_res = rule_2(x).detach()
                rule_3_res = rule_3(x).detach()
            elif args.dataset == "traffic":
                rule_1_res = rule_penalty(x).detach()
                rule_2_res = rule_payment(x).detach()
                rule_3_res = rule_amount(x).detach()
            x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
            x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
            x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
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

count_violated = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        if args.dataset == "sepsis":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = torch.logical_and(rule_crp_atb(x).detach(), rule_crp_100(x).detach())
        elif args.dataset == "bpi12":
            rule_1_res = f1(x).detach()
            f2_res = f2(x).detach()
            f3_res = f3(x).detach()
            f_resources_11169_res = f_resources_11169(x).detach()
            f_resources_10910_res = f_resources_10910(x).detach()
            rule_2_res = torch.logical_and(f2_res, f3_res).detach()
            rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        elif args.dataset == "bpi17":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = rule_3(x).detach()
        elif args.dataset == "traffic":
            rule_1_res = rule_penalty(x).detach()
            rule_2_res = rule_payment(x).detach()
            rule_3_res = rule_amount(x).detach()
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if args.dataset == "bpi12":
                if rule_1_res[i] == 1 and y[i] == 0:
                num_constraints == 1
                if predictions[i] == 0:
                    count_violated += 1
                if rule_2_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        count_violated += 1
                if rule_3_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        count_violated += 1
            elif args.dataset == "bpi17":
                if rule_1_res[i] == 1 and y[i] == 0:
                    num_constraints == 1
                    if predictions[i] == 1:
                        count_violated += 1
                if rule_2_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        count_violated += 1
                if rule_3_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        count_violated += 1
            else:
                if rule_1_res[i] == 1 and y[i] == 1:
                num_constraints == 1
                if predictions[i] == 1:
                    count_violated += 1
            if rule_2_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
                    count_violated += 1
            if rule_3_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] == 1:
                    count_violated += 1

metrics_lstm = []
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
print("Compliance:", count_violated / num_constraints)
metrics_lstm.append(count_violated / num_constraints)
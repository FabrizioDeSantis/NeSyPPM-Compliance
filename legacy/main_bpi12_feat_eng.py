import pandas as pd
from metrics import compute_metrics_fa
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.lstm import LSTMModelA, LSTMModel
from model.transformer import EventTransformer, EventTransformerA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import statistics
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
    parser.add_argument("--dataset_size", type=float, default=10, help="Size of the dataset (10%, 20%, 50%, 70%, 90%, 100%)")

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

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(data)

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

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
# lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

# RULES BPI12
f1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
f2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
f3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
f_resources_11169 = lambda x: (x[:, :240] == 48).any(dim=1)
f_resources_10910 = lambda x: (x[:, :240] == 21).any(dim=1)
# RULES BPI17
rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)
# RULES SEPSIS
high_lactic_acid = lambda x: (x[:, 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1)
check_sirs_criteria = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x])
rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)
# RULES TRAFFIC FINES
rule_penalty = lambda x: (x[:, :10] == 1).any(dim=1)
rule_payment = lambda x: (x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])
rule_amount = lambda x: (x[:, 90] > scalers["amount"].transform([[400]])[0][0])

lstm.train()
training_losses = []
validation_losses = []
max_val_f1 = 0.0
count_f1_val = 0
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        if dataset == "bpi12":
            rule_1_res = f1(x).detach()
            f2_res = f2(x).detach()
            f3_res = f3(x).detach()
            f_resources_11169_res = f_resources_11169(x).detach()
            f_resources_10910_res = f_resources_10910(x).detach()
            rule_2_res = torch.logical_and(f2_res, f3_res).detach()
            rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        elif dataset == "bpi17":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = rule_3(x).detach()
        elif dataset == "traffic":
            rule_1_res = rule_penalty(x).detach()
            rule_2_res = rule_payment(x).detach()
            rule_3_res = rule_amount(x).detach()
        elif dataset == "sepsis":
            rule_1_res = high_lactic_acid(x).detach()
            rule_2_res = check_sirs_criteria(x).detach()
            rule_3_res = torch.logical_and(rule_crp_atb(x), rule_crp_100(x)).detach()
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = x.to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output.squeeze(1).cpu(), y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
    training_losses.append(statistics.mean(train_losses))
    lstm.eval()
    with torch.no_grad():
        val_losses = []
        for enum, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                if dataset == "bpi12":
                    rule_1_res = f1(x).detach()
                    f2_res = f2(x).detach()
                    f3_res = f3(x).detach()
                    f_resources_11169_res = f_resources_11169(x).detach()
                    f_resources_10910_res = f_resources_10910(x).detach()
                    rule_2_res = torch.logical_and(f2_res, f3_res).detach()
                    rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
                elif dataset == "bpi17":
                    rule_1_res = rule_1(x).detach()
                    rule_2_res = rule_2(x).detach()
                    rule_3_res = rule_3(x).detach()
                elif dataset == "traffic":
                    rule_1_res = rule_penalty(x).detach()
                    rule_2_res = rule_payment(x).detach()
                    rule_3_res = rule_amount(x).detach()
                elif dataset == "sepsis":
                    rule_1_res = high_lactic_acid(x).detach()
                    rule_2_res = check_sirs_criteria(x).detach()
                    rule_3_res = torch.logical_and(rule_crp_atb(x), rule_crp_100(x)).detach()
                x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
                x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
                x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
                x = x.to(device)
                output = lstm(x)
                loss = criterion(output.squeeze(1).cpu(), y)
                val_losses.append(loss.item())
        print(f"Validation Loss: {statistics.mean(val_losses)}")
        validation_losses.append(statistics.mean(val_losses))
        _, f1score, _, _, _ = compute_metrics_fa(val_loader, lstm, device, "ltn", scalers, dataset)
        if f1score > max_val_f1:
            max_val_f1 = f1score
            torch.save(lstm.state_dict(), "best_model_lstm_2.pth")
            count_f1_val = 0
        else:
            count_f1_val += 1
            if count_f1_val >= 5:
                print("Early stopping")
                break
    lstm.train()

lstm.load_state_dict(torch.load("best_model_lstm_2.pth"))
lstm.eval()
y_pred = []
y_true = []
y_pred_corrected = []
compliance_lstm = 0
num_constraints = 0
count_violations = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        if dataset == "bpi12":
            rule_1_res = f1(x).detach()
            f2_res = f2(x).detach()
            f3_res = f3(x).detach()
            f_resources_11169_res = f_resources_11169(x).detach()
            f_resources_10910_res = f_resources_10910(x).detach()
            rule_2_res = torch.logical_and(f2_res, f3_res).detach()
            rule_3_res = torch.logical_or(f_resources_11169_res, f_resources_10910_res).detach()
        elif dataset == "bpi17":
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = rule_3(x).detach()
        elif dataset == "traffic":
            rule_1_res = rule_penalty(x).detach()
            rule_2_res = rule_payment(x).detach()
            rule_3_res = rule_amount(x).detach()
        elif dataset == "sepsis":
            rule_1_res = high_lactic_acid(x).detach()
            rule_2_res = check_sirs_criteria(x).detach()
            rule_3_res = torch.logical_and(rule_crp_atb(x), rule_crp_100(x)).detach()
        x = torch.cat([x, rule_1_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_2_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        x = torch.cat([x, rule_3_res.unsqueeze(1).repeat(1, max_prefix_length)], dim=1)
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_true.append(y[i].cpu())
            y_pred.append(predictions[i])
            if dataset == "bpi12":
                if rule_1_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] != 0:
                        count_violations += 1
                if rule_2_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] != 0:
                        count_violations += 1
                if rule_3_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] != 0:
                        count_violations += 1
            elif dataset == "bpi17":
                if rule_1_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
                if rule_2_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] != 0:
                        count_violations += 1
                if rule_3_res[i] == 1 and y[i] == 0:
                    num_constraints += 1
                    if predictions[i] != 0:
                        count_violations += 1
            elif dataset == "traffic":
                if rule_1_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
                if rule_2_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
                if rule_3_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
            elif dataset == "sepsis":
                if rule_1_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
                if rule_2_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1
                if rule_3_res[i] == 1 and y[i] == 1:
                    num_constraints += 1
                    if predictions[i] != 1:
                        count_violations += 1

print("Metrics LSTM without refinement:")
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
print("Metrics LSTM with refinement:")
accuracy = accuracy_score(y_true, y_pred_corrected)
metrics_lstm.append(accuracy)
print("Accuracy:", accuracy)
f1 = f1_score(y_true, y_pred_corrected, average='macro')
metrics_lstm.append(f1)
print("F1 Score:", f1)
precision = precision_score(y_true, y_pred_corrected, average='macro')
metrics_lstm.append(precision)
print("Precision:", precision)
recall = recall_score(y_true, y_pred_corrected, average='macro')
metrics_lstm.append(recall)
print("Recall:", recall)
print("Number of violations:", count_violations / num_constraints)
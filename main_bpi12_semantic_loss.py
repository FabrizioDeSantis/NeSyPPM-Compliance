import pandas as pd
from metrics import compute_metrics_fa
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.lstm import LSTMModelA, LSTMModel
from model.transformer import EventTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import statistics
from collections import defaultdict, Counter
from data import preprocess_bpi12
from data.dataset import NeSyDataset, ModelConfig
from model.semantic_loss import semantic_loss, semantic_loss_pos

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

print(device)

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

print(device)

lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
# lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

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

lstm.train()
training_losses = []
validation_losses = []
max_val_f1 = 0.0
count_f1_val = 0
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        rule_1_res = f1(x).detach()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        rules_res = torch.stack([rule_1_res, rule_2_res, rule_3_res], dim=1).to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = semantic_loss_pos(output.squeeze(1), y, rules_res, device)
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
                rule_1_res = f1(x).detach()
                f2_res = f2(x).detach()
                f3_res = f3(x).detach()
                f_resources_11169_res = f_resources_11169(x).detach()
                f_resources_10910_res = f_resources_10910(x).detach()
                rule_2_res = torch.logical_and(f2_res, f3_res).detach()
                rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
                rules_res = torch.stack([rule_1_res, rule_2_res, rule_3_res], dim=1).to(device)
                x = x.to(device)
                y = y.to(device)
                output = lstm(x)
                loss = semantic_loss_pos(output.squeeze(1), y, rules_res, device)
                val_losses.append(loss.item())
        print(f"Validation Loss: {statistics.mean(val_losses)}")
        validation_losses.append(statistics.mean(val_losses))
        _, f1score, _, _, _ = compute_metrics_fa(val_loader, lstm, device, "ltn", scalers, dataset)
        if f1score > max_val_f1:
            max_val_f1 = f1score
            torch.save(lstm.state_dict(), "best_model_lstm.pth")
            count_f1_val = 0
        else:
            count_f1_val += 1
            if count_f1_val >= 5:
                print("Early stopping")
                break
    lstm.train()

lstm.load_state_dict(torch.load("best_model_lstm.pth"))
lstm.eval()
y_pred = []
y_true = []
y_pred_corrected = []
rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)
compliance_lstm = 0
num_constraints = 0
count_violations = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        rule_amount_1_res = rule_amount_1(x).detach().cpu().numpy()
        rule_amount_2_res = rule_amount_2(x).detach().cpu().numpy()
        rule_amount_3_res = rule_amount_3(x).detach().cpu().numpy()
        rule_resource_1_res = rule_resource_1(x).detach().cpu().numpy()
        rule_resource_2_res = rule_resource_2(x).detach().cpu().numpy()
        rule_1_res = f1(x).detach()
        f2_res = f2(x).detach()
        f3_res = f3(x).detach()
        f_resources_11169_res = f_resources_11169(x).detach()
        f_resources_10910_res = f_resources_10910(x).detach()
        rule_2_res = torch.logical_and(f2_res, f3_res).detach()
        rule_3_res = torch.logical_and(f_resources_11169_res, f_resources_10910_res).detach()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_true.append(y[i].cpu())
            y_pred.append(predictions[i])
            if rule_amount_1_res[i] == 1 and y[i] == 0:
                if predictions[i] == 1:
                    count_violations += 1
                #######
                predictions[i] = 0
                #######
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_amount_2_res[i] == 1 and rule_amount_3_res[i] == 1 and y[i] == 0:
                if predictions[i] == 1:
                    count_violations += 1
                #######
                predictions[i] = 0
                #######
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_resource_1_res[i] == 1 and y[i] == 0:
                if predictions[i] == 1:
                    count_violations += 1
                #######
                predictions[i] =0
                #######
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            if rule_resource_2_res[i] == 1 and y[i] == 0:
                if predictions[i] == 1:
                    count_violations += 1
                #######
                predictions[i] =0
                #######
                num_constraints += 1
                if predictions[i] == 0:
                    compliance_lstm += 1
            y_pred_corrected.append(predictions[i])

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
print("Compliance:", compliance_lstm / num_constraints)
metrics_lstm.append(compliance_lstm / num_constraints)
print("Number of violations:", count_violations / num_constraints)
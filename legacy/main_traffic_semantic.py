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
from data import preprocess_traffic
from data.dataset import NeSyDataset, ModelConfig
from model.semantic_loss import semantic_loss, semantic_loss_pos
from model.transformer import EventTransformer
import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "traffic_fines"

classes = ["Repaid", "Send for credit collection"]

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
    parser.add_argument("--num_epochs_nesy", type=int, default=15, help="Number of epochs for training LTN model")
    parser.add_argument("--dataset", type=str, default="sepsis", help="Dataset to use")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")

    return parser.parse_args()

args = get_args()

dataset = "traffic_fines"
max_prefix_length = 10

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

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_traffic.preprocess_eventlog(data)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)
numerical_features = ["expense", "amount", "paymentAmount"]

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

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
        rule_1_res = rule_penalty(x).detach()
        rule_2_res = rule_payment(x).detach()
        rule_3_res = rule_amount(x).detach()
        rules_pos_res = torch.stack([rule_1_res, rule_2_res, rule_3_res], dim=1).to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = semantic_loss_pos(output.squeeze(1), y, rules_pos_res, device)
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
            rule_1_res = rule_penalty(x).detach()
            rule_2_res = rule_payment(x).detach()
            rule_3_res = rule_amount(x).detach()
            rules_pos_res = torch.stack([rule_1_res, rule_2_res, rule_3_res], dim=1).to(device)
            output = lstm(x)
            loss = semantic_loss_pos(output.squeeze(1), y, rules_pos_res, device)
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
        rule_penalty_res = rule_penalty(x).detach().cpu().numpy()
        rule_payment_res = rule_payment(x).detach().cpu().numpy()
        rule_amount_res = rule_amount(x).detach().cpu().numpy()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if rule_penalty_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] != 1:
                    count_violated += 1
            if rule_payment_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] != 1:
                    count_violated += 1
            if rule_amount_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] != 1:
                    count_violated += 1

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
print(count_r2)
print(num_constraints)
print("Violated constraints:", count_violated / num_constraints)
metrics_lstm.append(count_violated / num_constraints)
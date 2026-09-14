import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.lstm import LSTMModel, LSTMModelA
from model.transformer import EventTransformer, EventTransformerA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_metrics, compute_metrics_fa
from collections import defaultdict, Counter
from data import preprocess_bpi17
from data.dataset import NeSyDataset, ModelConfig
import matplotlib.pyplot as plt
import seaborn as sns
from model.semantic_loss import semantic_loss, semantic_loss_pos
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
    parser.add_argument("--dataset_size", type=float, default=10, help="Size of the dataset (10%, 20%, 50%, 70%, 90%, 100%)")
    parser.add_argument("--feat", type=bool, default=True, help="Use feature engineering")
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

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_bpi17.preprocess_eventlog(data)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

numerical_features = ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount"]

# lstm = LSTMModelA(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
# lstm = EventTransformerA(vocab_sizes, config, feature_names, numerical_features, model_dim=128, num_classes=1, max_len=max_prefix_length).to(device)
lstm = EventTransformer(vocab_sizes, config, feature_names, numerical_features, model_dim=64, num_classes=1, max_len=max_prefix_length).to(device)
# lstm = LSTMModel(vocab_sizes, config, 1, feature_names, numerical_features).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
criterion = torch.nn.BCELoss()

rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)

lstm.train()
training_losses = []
validation_losses = []
for epoch in range(config.num_epochs):
    train_losses = []
    for enum, (x, y) in enumerate(train_loader):
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        rules_pos_res = torch.stack([rule_1_res], dim=1).to(device)
        rules_neg_res = torch.stack([rule_2_res, rule_3_res], dim=1).to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = lstm(x)
        loss = semantic_loss(output.squeeze(1), y, rules_pos_res, rules_neg_res)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
    training_losses.append(statistics.mean(train_losses))
    lstm.eval()
    val_losses = []
    for enum, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            rule_1_res = rule_1(x).detach()
            rule_2_res = rule_2(x).detach()
            rule_3_res = rule_3(x).detach()
            rules_pos_res = torch.stack([rule_1_res], dim=1).to(device)
            rules_neg_res = torch.stack([rule_2_res, rule_3_res], dim=1).to(device)
            x = x.to(device)
            y = y.to(device)
            output = lstm(x)
            loss = semantic_loss(output.squeeze(1), y, rules_pos_res, rules_neg_res)
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
violated_constraints = 0
num_constraints = 0
for enum, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        x = x.to(device)
        # Apply the rule to the input data
        rule_1_res = rule_1(x).detach()
        rule_2_res = rule_2(x).detach()
        rule_3_res = rule_3(x).detach()
        outputs = lstm(x).detach().cpu().numpy()
        predictions = np.where(outputs > 0.5, 1., 0.).flatten()
        for i in range(len(y)):
            y_pred.append(predictions[i])
            y_true.append(y[i].cpu())
            if rule_1_res[i] == 1 and y[i] == 1:
                num_constraints += 1
                if predictions[i] != 1:
                    violated_constraints += 1
            if rule_2_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] != 0:
                    violated_constraints += 1
            if rule_3_res[i] == 1 and y[i] == 0:
                num_constraints += 1
                if predictions[i] != 0:
                    violated_constraints += 1
            
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
print("Violated constraints:", violated_constraints / num_constraints)
metrics_lstm.append(violated_constraints / num_constraints)
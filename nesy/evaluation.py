'''Test metrics, computed in the same way for every configuration.'''
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

METRICS = ["accuracy", "f1", "precision", "recall", "compliance"]


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


@torch.no_grad()
def evaluate(model, loader, knowledge, model_input, device):
    '''
    accuracy, macro precision / recall / F1 of the predictions (threshold 0.5) and
        compliance: among the (rule, prefix) pairs where a logical rule fires and the true label agrees with it,
                    the fraction where the prediction agrees too (see nesy/knowledge.py), also per rule
    model_input: x -> input of the model (nesy/training.ModelInput)
    '''
    model.eval()
    y_true, y_pred, truth = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predictions = (model(model_input(x)).squeeze(1) > 0.5).long()
        y_true.append(y.long().cpu())
        y_pred.append(predictions.cpu())
        truth.append(knowledge.truth(x).cpu())
    y_true, y_pred, truth = torch.cat(y_true), torch.cat(y_pred), torch.cat(truth)

    rule_labels = knowledge.labels()
    applicable = truth & (y_true.unsqueeze(1) == rule_labels)
    satisfied = applicable & (y_pred.unsqueeze(1) == rule_labels)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "compliance": ratio(satisfied.sum(), applicable.sum()),
        "rule_compliance": {rule.name: ratio(satisfied[:, k].sum(), applicable[:, k].sum())
                            for k, rule in enumerate(knowledge.feature_rules)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics

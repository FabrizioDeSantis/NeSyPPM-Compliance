import torch
import torch.nn.functional as F


def semantic_loss(probs, labels, positive_triggers, negative_triggers, lambda_sem=0.1, epsilon=1e-7):
    '''
    Binary cross-entropy + lambda_sem * semantic term.

    probs, labels: (B,) predicted P(label = 1) and true labels
    positive_triggers / negative_triggers: (B, K+) / (B, K-) truth values of the rules implying label 1 /
        label 0 (K+ or K- can be 0)
    The semantic term is the negative log-likelihood of the implied label, for every prefix where at least
    one rule fires, averaged over the batch.
    '''
    ce_loss = F.binary_cross_entropy(probs, labels)
    fires_pos = positive_triggers.any(dim=1).float()
    fires_neg = negative_triggers.any(dim=1).float()
    pos_loss = -torch.log(probs.clamp(epsilon, 1.0)) * fires_pos
    neg_loss = -torch.log((1.0 - probs).clamp(epsilon, 1.0)) * fires_neg
    return ce_loss + lambda_sem * (pos_loss + neg_loss).mean()

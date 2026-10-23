import torch
import torch.nn.functional as F

def semantic_loss(propbs_P, labels_P, positive_triggers, negative_triggers, lambda_sem=0.1):
    epsilon = 1e-7
    
    ce_loss = F.binary_cross_entropy(propbs_P, labels_P)

    pos_mask, _ = torch.max(positive_triggers, dim=1, keepdim=True)
    pos_mask = pos_mask.to(propbs_P.device)

    pos_loss = -torch.log(torch.clamp(propbs_P, epsilon, 1.0)) * pos_mask

    neg_mask, _ = torch.max(negative_triggers, dim=1, keepdim=True)
    neg_mask = neg_mask.to(propbs_P.device)
    
    neg_loss = -torch.log(torch.clamp(1.0 - propbs_P, epsilon, 1.0)) * neg_mask

    semantic_loss = torch.mean(pos_loss + neg_loss)

    total_loss = ce_loss + lambda_sem * semantic_loss

    return total_loss

def semantic_loss_pos(propbs_P, labels_P, positive_triggers, device, lambda_sem=0.1):
    epsilon = 1e-7
    
    ce_loss = F.binary_cross_entropy(propbs_P, labels_P)

    pos_mask, _ = torch.max(positive_triggers, dim=1, keepdim=True)
    pos_mask = pos_mask.to(device)

    pos_loss = -torch.log(torch.clamp(propbs_P, epsilon, 1.0)) * pos_mask

    sem_loss = torch.mean(pos_loss)

    total_loss = ce_loss + lambda_sem * sem_loss

    return total_loss
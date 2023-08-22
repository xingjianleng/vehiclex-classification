import torch
import torch.nn.functional as F

    
def focal_loss(input, target, gamma=2, weight=None):
    log_prob = F.log_softmax(input, dim=1)
    prob = torch.exp(log_prob)
    log_prob = (1 - prob) ** gamma * log_prob
    loss = F.nll_loss(log_prob, target, weight)
    return loss

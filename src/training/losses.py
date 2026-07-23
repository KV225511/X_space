import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification tasks.
    Down-weights well-classified examples to focus on hard examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute standard cross entropy loss (the base penalty)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate the probability that the model guessed correctly
        pt = torch.exp(-ce_loss)
        
        # Multiply the base penalty by the Focal factor to punish bad guesses on rare classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class OODEnergyLoss(nn.Module):
    """
    Energy-based Out-Of-Distribution (OOD) loss.
    Calculates if the input looks completely foreign to the model.
    """
    def __init__(self, temperature=1.0):
        super(OODEnergyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        # Calculate the energy score. High energy = Anomaly/OOD.
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy.mean()
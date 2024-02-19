import torch.nn as nn


class WeightedCELoss(nn.Module):
    def __init__(self, fine_weight=2, coarse_weight=1):
        super(WeightedCELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.fine_weight = fine_weight
        self.coarse_weight = coarse_weight

    def forward(self, fine_pred, coarse_pred, targets):
        fine_loss = self.celoss(fine_pred, targets)
        coarse_loss = self.celoss(coarse_pred, targets)
        tot_loss = self.fine_weight * fine_loss + self.coarse_weight * coarse_loss
        return tot_loss, fine_loss, coarse_loss

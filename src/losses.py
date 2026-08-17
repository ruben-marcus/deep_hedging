import torch
from torch import nn


class CVaRLoss(nn.Module):
    def __init(self, alpha=0.95):
        super().__init__()

        self.alpha = alpha
        self.eta = nn.Parameter(torch.tensor(0.0))

    def forward(self, pnl):
        loss = -pnl
        tail = torch.relu(loss - self.eta)

        return self.eta + tail.mean() / (1.0 - self.alpha)

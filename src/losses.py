import math
import torch
from torch import nn


class MSEHedgingLoss(nn.Module):
    def forward(self, pnl):
        return torch.mean(pnl**2)


class CVaRLoss(nn.Module):
    def __init__(self, alpha=0.95):
        super().__init__()

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        self.alpha = alpha

        # trainable VaR threshold
        self.eta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, pnl):
        loss = -pnl
        tail = torch.relu(loss - self.eta)

        return self.eta + tail.mean() / (1.0 - self.alpha)


class EntropicRiskLoss(nn.Module):
    def __init__(self, risk_aversion=0.1):
        super().__init__()

        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")

        self.risk_aversion = risk_aversion

    def forward(self, pnl):
        lam = self.risk_aversion
        z = -lam * pnl

        # numerically stable version of
        # (1 / lambda) * log(mean(exp(-lambda * pnl)))
        return (
            torch.logsumexp(z, dim=0) - math.log(pnl.numel())
        ) / lam

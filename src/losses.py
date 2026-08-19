import math
import torch
from torch import nn

# risk objectives applied to terminal hedging pnl, shape (batch_size,)
# lower means better hedge


class MSEHedgingLoss(nn.Module):
    """mean squared hedging error, penalizes gains and losses symmetrically"""

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        return torch.mean(pnl**2)


class CVaRLoss(nn.Module):
    """
    conditional value at risk of the loss -pnl
    mean of the worst (1 - alpha) tail (only penalizes downside)
    uses the rockafellar-uryasev form: 
    eta is a learned parameter, passed to optimizer alongside model params
    """

    def __init__(self, alpha: float = 0.95):
        super().__init__()

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        self.alpha = alpha

        # trainable VaR threshold
        self.eta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        loss = -pnl
        tail = torch.relu(loss - self.eta)

        return self.eta + tail.mean() / (1.0 - self.alpha)


class EntropicRiskLoss(nn.Module):
    """
    entropic risk measure (exponential utility)

    larger risk_aversion weighs bad outcomes more heavily
    risk_aversion -> 0 means that objective -> -E[pnl]
    """

    def __init__(self, risk_aversion: float = 0.1):
        super().__init__()

        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")

        self.risk_aversion = risk_aversion

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        lam = self.risk_aversion
        z = -lam * pnl

        # numerically stable version of
        # (1 / lambda) * log(mean(exp(-lambda * pnl)))
        return (
            torch.logsumexp(z, dim=0) - math.log(pnl.numel())
        ) / lam

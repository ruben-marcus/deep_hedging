import numpy as np
import pytest
import torch

from losses import MSEHedgingLoss, CVaRLoss, EntropicRiskLoss
from metrics import empirical_cvar


@pytest.fixture
def pnl():
    return torch.tensor(
        np.random.default_rng(11).standard_normal(20_000) * 2.0,
        dtype=torch.float64,
    )


def test_cvar_loss_at_optimal_eta_equals_empirical_cvar(pnl):
    """
    rockafeller-uryasev objective minimized at eta = VaR,
    where its value is the CVaR
    """

    alpha = 0.95
    loss_fn = CVaRLoss(alpha=alpha).double()

    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.05)

    for _ in range(2_000):
        optimizer.zero_grad()
        loss_fn(pnl).backward()
        optimizer.step()

    assert float(loss_fn(pnl).detach()) == pytest.approx(
        empirical_cvar(pnl, alpha=alpha), rel=0.01
    )

    assert float(loss_fn.eta.detach()) == pytest.approx(
        float(np.quantile(-pnl.numpy(), alpha)), rel=0.05
    )


def test_entropic_is_numerically_stable():
    """logsumexp form ensures large losses don't overflow exp(-lambda * pnl)"""

    pnl = torch.tensor([-5_000.0, 0.0, 5_000.0], dtype=torch.float64)

    value = EntropicRiskLoss(risk_aversion=1.0)(pnl)

    assert torch.isfinite(value)
    assert float(torch.exp(-1.0 * pnl).mean()
                 ) == float("inf")  # naive overflows


@pytest.mark.parametrize("risk_aversion", [0.0, -1.0])
def test_entropic_rejects_bad_risk_aversion(risk_aversion):
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        EntropicRiskLoss(risk_aversion=risk_aversion)

import numpy as np
import pytest
from scipy.stats import norm

from metrics import empirical_cvar, hedge_metrics, pnl_metrics


def test_cvar_exact_tail_by_hand():
    """losses are 0..99, so worst 5% is {95..99}, mean is 97"""

    pnl = -np.arange(100.0)

    assert empirical_cvar(pnl, alpha=0.95) == pytest.approx(97.0)
    assert empirical_cvar(pnl, alpha=0.99) == pytest.approx(99.0)
    assert empirical_cvar(pnl, alpha=0.90) == pytest.approx(94.5)


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_cvar_matches_analytic_normal(alpha):
    """expected shortfall of standard normal is phi(z_alpha) / (1 - alpha)"""

    pnl = -np.random.default_rng(12).standard_normal(1_000_000)

    analytic = norm.pdf(norm.ppf(alpha)) / (1.0 - alpha)

    assert empirical_cvar(pnl, alpha=alpha) == pytest.approx(
        analytic,
        rel=0.01
    )


def test_cvar_increase_with_alpha():
    """deeper tail is worse"""

    pnl = np.random.default_rng(13).standard_normal(100_000)

    alphas = [0.90, 0.95, 0.99, 0.995]
    values = [empirical_cvar(pnl, alpha=a) for a in alphas]

    assert values == sorted(values)


def test_pnl_metrics_keys_and_values():
    pnl = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    metrics = pnl_metrics(pnl, cvar_alphas=(0.95,))

    assert set(metrics) == {
        "mean_pnl",
        "std_pnl",
        "rmse",
        "q05",
        "q01",
        "cvar95_loss"
    }

    assert metrics["mean_pnl"] == pytest.approx(0.0)
    assert metrics["std_pnl"] == pytest.approx(np.std(pnl, ddof=1))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(pnl**2)))
    assert all(isinstance(v, float) for v in metrics.values())


def test_hedge_metrics_adds_cost_and_turnover():
    result = {
        "pnl": np.array([1.0, -1.0, 2.0, -2.0]),
        "transaction_costs": np.array([0.1, 0.2, 0.3, 0.4]),
        "share_turnover": np.array([1.0, 2.0, 3.0, 4.0]),
        "notional_turnover": np.array([10.0, 20.0, 30.0, 40.0]),
    }

    metrics = hedge_metrics(result)

    assert metrics["mean_tc"] == pytest.approx(0.25)
    assert metrics["mean_share_turnover"] == pytest.approx(2.5)
    assert metrics["mean_notional_turnover"] == pytest.approx(25.0)

    assert "mean_pnl" in metrics and "cvar95_loss" in metrics

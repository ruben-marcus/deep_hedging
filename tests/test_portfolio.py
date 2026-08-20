import numpy as np
import torch

from derivatives.european_call import black_scholes_call, european_call_payoff
from models.gbm import simulate_gbm
from portfolio import build_state, run_hedge, run_hedge_torch
from strategies.delta_hedge import run_delta_hedge


def test_build_state_column_layout():
    """
    pins feature contract that run_hedge_torch and policy plots depend on:
    log moneyness, scaled variance, time to maturity, shares
    """

    S = torch.tensor([80.0, 100.0, 125.0])
    v = torch.tensor([0.01, 0.04, 0.09])
    shares = torch.tensor([-0.5, 0.0, 1.5])
    K, T, t, scale = 100.0, 2.0, 0.5, 0.04

    state = build_state(
        S=S,
        v=v,
        t=t,
        shares=shares,
        K=K,
        T=T,
        variance_scale=scale,
    )

    assert state.shape == (3, 4)

    assert torch.allclose(state[:, 0], torch.log(S / K))
    assert torch.allclose(state[:, 1], v / scale)
    assert torch.allclose(state[:, 2], torch.full_like(S, (T - t) / T))
    assert torch.allclose(state[:, 3], shares)


def test_cash_accrues_at_risk_free_rate(market):
    """trade nothing, expect cash to compound over one maturity"""

    r, T, premium = 0.05, 1.0, 10.0

    paths = simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=T,
        n_steps=10,
        n_paths=5,
        seed=14,
    ).spot

    result = run_hedge(
        stock_paths=paths,
        initial_premium=premium,
        position_fn=lambda S, t: np.zeros(len(S)),
        payoff_fn=lambda S_T: np.zeros(len(S_T)),
        T=T,
        r=r,
    )

    assert np.allclose(result["terminal_cash"], premium * np.exp(r * T))
    assert np.allclose(result["terminal_shares"], 0.0)


def test_self_financing_identity(market):
    """cash moves by exactly minus traded notional and costs"""

    T, cost_rate = 1.0, 0.001

    paths = simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=T,
        n_steps=10,
        n_paths=200,
        seed=15,
    ).spot

    result = run_delta_hedge(
        stock_paths=paths,
        K=market["K"],
        sigma=market["sigma"],
        T=T,
        r=0.0,
        transaction_cost_rate=cost_rate,
        record_history=True,
    )

    traded_notional = (result["trades"] * paths[:, :-1]).sum(axis=1)
    expected_cash = result["initial_premium"] - (
        traded_notional + result["costs"].sum(axis=1)
    )

    assert np.allclose(result["terminal_cash"], expected_cash)


def test_delta_hedge_error_shrinks_with_rebalancing(market):
    """
    hedging error should decrease as step frequency increases
    this should fall roughly like 1/sqrt(n_steps)
    """

    sigma, T, K = market["sigma"], market["T"], market["K"]
    stds = []

    for n_steps in [10, 100, 1000]:
        paths = simulate_gbm(
            S0=market["S0"],
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=5_000,
            seed=16,
        ).spot

        result = run_delta_hedge(
            stock_paths=paths,
            K=K,
            sigma=sigma,
            T=T,
        )

        assert abs(result["pnl"].mean()) < 0.1
        stds.append(result["pnl"].std(ddof=1))

    assert stds[0] > stds[1] > stds[2]
    assert stds[2] < 0.5

    # each 10x step increase should reduce error by about sqrt(10) ~3.16
    for coarse, fine in zip(stds, stds[1:]):
        assert 2.0 < coarse / fine < 5.0


def test_torch_matches_numpy_for_same_policy(market, bs_policy):
    """ensure both runners agree with same black-scholes delta"""

    sigma, K, T = market["sigma"], market["K"], market["T"]

    sim = simulate_gbm(
        S0=market["S0"],
        sigma=sigma,
        T=T,
        n_steps=30,
        n_paths=1_000,
        seed=17,
    )

    premium = black_scholes_call(
        S0=market["S0"],
        K=K,
        sigma=sigma,
        T=T,
    )

    numpy_result = run_delta_hedge(
        stock_paths=sim.spot,
        K=K,
        sigma=sigma,
        T=T,
        transaction_cost_rate=0.001,
    )

    torch_result = run_hedge_torch(
        model=bs_policy,
        spot_paths=torch.tensor(sim.spot, dtype=torch.float64),
        variance_paths=torch.tensor(
            np.asarray(sim.variance),
            dtype=torch.float64
        ),
        K=K,
        T=T,
        premium=float(premium),
        transaction_cost_rate=0.001,
    )

    assert np.allclose(
        torch_result["pnl"].numpy(),
        numpy_result["pnl"],
        atol=1e-8,
    )

    assert np.allclose(
        torch_result["share_turnover"].numpy(),
        numpy_result["share_turnover"],
        atol=1e-8,
    )


def test_torch_is_differentiable(market):
    """gradients must reach the policy"""

    model = torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Flatten(0))

    sim = simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=1.0,
        n_steps=10,
        n_paths=100,
        seed=18,
    )

    result = run_hedge_torch(
        model=model,
        spot_paths=torch.tensor(sim.spot, dtype=torch.float32),
        variance_paths=torch.tensor(np.asarray(
            sim.variance), dtype=torch.float32),
        K=market["K"],
        T=1.0,
        premium=8.0,
        transaction_cost_rate=0.001,
    )

    result["pnl"].pow(2).mean().backward()

    grads = [p.grad for p in model.parameters()]

    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

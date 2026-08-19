from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


# eventually change to return same diagnostics as run_hedge_torch
def run_hedge(
    stock_paths: npt.ArrayLike,
    initial_premium: npt.ArrayLike,
    position_fn: Callable[[np.ndarray, float], npt.ArrayLike],
    payoff_fn: Callable[[np.ndarray], npt.ArrayLike],
    T: float,
    r: float = 0.0,
    transaction_cost_rate: float = 0.0,
    record_history: bool = False,
) -> dict[str, np.ndarray]:
    """
    run a self-financing hedge across many stock paths

    book is short one derivative: collect initial_premium as cash,
    trade shares at each step, and settle payoff_fn at maturity

    parameters
    ----------
    stock_paths : array
        shape (n_paths, n_steps + 1)

    initial_premium : float or array
        cash received at t=0, broadcast to (n_paths,)

    position_fn : callable
        function(S, t) -> desired number of shares

    payoff_fn : callable
        function(S_T) -> terminal derivative payoff

    T : float
        maturity in years

    transaction_cost_rate : float
        cost charged as rate * |trade| * S

    record_history : bool
        result includes per-step positions, trades, costs, and cash

    returns
    -------
    dict of arrays, each shape (n_paths,)
        pnl                 terminal portfolio minus payoff (hedging error)
        payoff              derivative payoff at maturity
        terminal_portfolio  cash + shares * S_T
        terminal_cash       cash left after settling
        terminal_shares     shares held into maturity
        transaction_costs   total costs paid
        share_turnover      sum of |trade| in shares
        notional_turnover   sum of |trade| * S
        initial_premium     premium per path

        with record_history: includes positions, trades, costs, and cash_history,
        each shape (n_paths, n_steps)
    """

    stock_paths = np.asarray(stock_paths, dtype=float)

    if stock_paths.ndim != 2:
        raise ValueError(
            "stock_paths must have shape (n_paths, n_steps + 1)"
        )

    n_paths, n_observations = stock_paths.shape
    n_steps = n_observations - 1

    if n_steps < 1:
        raise ValueError("at least one hedging interval is required")

    dt = T / n_steps

    # initial wealth = premium received from short call
    premium = np.broadcast_to(
        np.asarray(initial_premium, dtype=float),
        (n_paths,),
    ).copy()

    cash = premium.copy()
    shares = np.zeros(n_paths, dtype=float)
    total_cost = np.zeros(n_paths, dtype=float)
    share_turnover = np.zeros(n_paths, dtype=float)
    notional_turnover = np.zeros(n_paths, dtype=float)

    if record_history:
        positions = np.empty((n_paths, n_steps))
        trades = np.empty((n_paths, n_steps))
        costs = np.empty((n_paths, n_steps))
        cash_history = np.empty((n_paths, n_steps))

    for i in range(n_steps):
        # no interest at i = 0
        if i > 0:
            cash *= np.exp(r * dt)

        S_i = stock_paths[:, i]
        t_i = i * dt

        target_shares = np.asarray(position_fn(S_i, t_i), dtype=float)
        target_shares = np.broadcast_to(target_shares, (n_paths,)).copy()

        trade = target_shares - shares
        abs_trade = np.abs(trade)

        share_turnover += abs_trade

        traded_notional = S_i * abs_trade
        notional_turnover += traded_notional

        transaction_cost = transaction_cost_rate * traded_notional
        total_cost += transaction_cost

        # self-financing
        cash -= trade * S_i
        cash -= transaction_cost

        shares = target_shares

        if record_history:
            positions[:, i] = shares
            trades[:, i] = trade
            costs[:, i] = transaction_cost
            cash_history[:, i] = cash

    # cash earns interest during final interval
    cash *= np.exp(r * dt)

    S_T = stock_paths[:, -1]

    terminal_portfolio = cash + shares * S_T
    payoff = np.asarray(payoff_fn(S_T), dtype=float)
    pnl = terminal_portfolio - payoff

    result = {
        "pnl": pnl,
        "payoff": payoff,
        "terminal_portfolio": terminal_portfolio,
        "terminal_cash": cash,
        "terminal_shares": shares,
        "transaction_costs": total_cost,
        "share_turnover": share_turnover,
        "notional_turnover": notional_turnover,
        "initial_premium": premium,
    }

    if record_history:
        result["positions"] = positions
        result["trades"] = trades
        result["costs"] = costs
        result["cash_history"] = cash_history

    return result


def build_state(
    S: torch.Tensor,
    v: torch.Tensor | float,
    t: float,
    shares: torch.Tensor | float,
    K: float,
    T: float,
    variance_scale: float = 0.04,
) -> torch.Tensor:
    """
    hedging state fed to policy, shape (batch_size, 4)

    features:
        log moneyness, scaled variance, time to maturity, current shares

    run_hedge_torch and notebook policy plots both go through here

    v and shares may be scalars, broadcast to shape of S
    """

    def feature(x: torch.Tensor | float) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        else:
            return torch.full_like(S, float(x))

    log_moneyness = torch.log(S / K)
    variance_feature = feature(v) / variance_scale
    tau_feature = torch.full_like(S, (T - t) / T)

    return torch.stack([
        log_moneyness,
        variance_feature,
        tau_feature,
        feature(shares),
    ], dim=1)


def run_hedge_torch(
    model: nn.Module,
    spot_paths: torch.Tensor,
    variance_paths: torch.Tensor,
    K: float,
    T: float,
    premium: float,
    transaction_cost_rate: float = 0.0,
    r: float = 0.0,
    variance_scale: float = 0.04,
) -> dict[str, torch.Tensor]:
    """
    differentiable version of run_hedge for short european call
    positions come from the model

    parameters
    ----------
    model : nn.Module
        maps state to share position (see NeuralHedge)

    spot_paths : tensor
        shape (batch_size, n_steps + 1)

    variance_paths : tensor
        shape (batch_size, n_steps + 1)
        constant tensor for gbm

    premium : float
        cash received at t=0, same for every path

    variance_scale : float
        divides variance feature to keep it near 1

    returns
    -------
    dict of tensors, each shape (batch_size,)
        pnl, transaction_costs, share_turnover, notional_turnover
    """

    batch_size, n_observations = spot_paths.shape

    n_steps = n_observations - 1

    dt = T / n_steps

    device = spot_paths.device
    dtype = spot_paths.dtype

    cash = torch.full(
        (batch_size,),
        float(premium),
        device=device,
        dtype=dtype,
    )

    shares = torch.zeros(
        batch_size,
        device=device,
        dtype=dtype,
    )

    share_turnover = torch.zeros_like(shares)
    notional_turnover = torch.zeros_like(shares)
    total_cost = torch.zeros_like(shares)

    growth = torch.exp(
        torch.tensor(r * dt, device=device, dtype=dtype)
    )

    for i in range(n_steps):
        # no interest at i = 0
        if i > 0:
            cash = cash * growth

        S_t = spot_paths[:, i]
        v_t = variance_paths[:, i]

        t = i * dt

        state = build_state(
            S=S_t,
            v=v_t,
            t=t,
            shares=shares,
            K=K,
            T=T,
            variance_scale=variance_scale,
        )

        target_shares = model(state)

        trade = target_shares - shares
        abs_trade = torch.abs(trade)

        share_turnover = share_turnover + abs_trade

        traded_notional = S_t * abs_trade
        notional_turnover = notional_turnover + traded_notional

        transaction_cost = transaction_cost_rate * traded_notional
        total_cost = total_cost + transaction_cost

        cash = cash - trade * S_t - transaction_cost

        shares = target_shares

    # final cash accrual
    cash = cash * growth

    S_T = spot_paths[:, -1]

    terminal_portfolio = cash + shares * S_T
    payoff = torch.relu(S_T - K)
    pnl = terminal_portfolio - payoff

    return {
        "pnl": pnl,
        "transaction_costs": total_cost,
        "share_turnover": share_turnover,
        "notional_turnover": notional_turnover,
    }

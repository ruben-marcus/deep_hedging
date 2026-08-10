import numpy as np


def run_self_financing_hedge(
    stock_paths,
    initial_premium,
    position_fn,
    payoff_fn,
    T,
    r=0.0,
    transaction_cost_rate=0.0,
    record_history=False,
):
    """
    run a self-financing hedge across many stock paths

    stock_paths:
        shape: (n_paths, n_steps + 1)

    position_fn:
        function(S, t) -> desired number of shares

    payoff_fn:
        function(S_T) -> terminal derivative payoff
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
    shares = np.zeros(n_paths)

    cumulative_costs = np.zeros(n_paths)

    if record_history:
        positions = np.empty((n_paths, n_steps))
        trades = np.empty((n_paths, n_steps))
        costs = np.empty((n_paths, n_steps))
        cash_history = np.empty((n_paths, n_steps))

    for i in range(n_steps):
        if i > 0:
            cash *= np.exp(r * dt)

        S_i = stock_paths[:, i]
        t_i = i * dt

        target_shares = np.asarray(position_fn(S_i, t_i), dtype=float)
        target_shares = np.broadcast_to(target_shares, (n_paths,)).copy()

        trade = target_shares - shares
        transaction_cost = transaction_cost_rate * S_i * np.abs(trade)

        cash -= trade * S_i
        cash -= transaction_cost

        cumulative_costs += transaction_cost

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
        "transaction_costs": cumulative_costs,
        "initial_premium": premium
    }

    if record_history:
        result["positions"] = positions
        result["trades"] = trades
        result["costs"] = costs
        result["cash_history"] = cash_history

    return result

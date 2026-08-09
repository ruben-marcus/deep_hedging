import numpy as np
import numpy.typing as npt


def delta_hedge_path(
    stock_path: npt.NDArray[np.float64] | list[float],
    K: float,
    sigma: float,
    T: float,
    r: float = 0.0
):
    """
    delta hedge one short european call along one stock path

    parameters
    ----------
    stock_path:  array/list
        stock prices: [S_0, S_1, ..., S_N]

    K : float
        strike price

    sigma : float
        annualized volatility

    T : float
        maturity time in years

    r : float
        risk-free rate

    returns
    -------
    result : dict
        contains hedge positions, cash balances, terminal payoff, portfolio value, hedging P&L
    """

    stock_path = np.asarray(stock_path, dtype=float)

    if stock_path.ndim != 1:
        raise ValueError("stock_path must be one-dimensional")

    if len(stock_path) < 2:
        raise ValueError("stock_path must contain at least two prices")

    n_steps = len(stock_path) - 1

    dt = T / n_steps

    times = np.linspace(0.0, T, n_steps + 1)

    # price received for selling the call
    option_premium = black_scholes_call(
        S=stock_path[0],
        K=K,
        sigma=sigma,
        T=T,
        t=0.0,
        r=r
    )

    deltas = np.empty(n_steps)
    trades = np.empty(n_steps)
    cash_balances = np.empty(n_steps)

    cash = option_premium
    previous_delta = 0.0

    # hedge at t_0, ... t_{N - 1}
    # do not rebalance at expiration
    for i in range(n_steps):
        S = stock_path[i]
        t = times[i]

        # existing cash earns the risk-free rate
        if i > 0:
            cash *= math.exp(r * dt)

        new_delta = black_scholes_delta(
            S=S,
            K=K,
            sigma=sigma,
            T=T,
            t=t,
            r=r
        )

        # number of shares to buy/sell
        trade = new_delta - previous_delta

        cash -= trade * S

        deltas[i] = new_delta
        trades[i] = trade
        cash_balances[i] = cash

        previous_delta = new_delta

    # cash accrues over final interval
    cash *= math.exp(r * dt)

    S_T = stock_path[-1]
    stock_value = previous_delta * S_T

    terminal_portfolio = stock_value + cash

    payoff = max(S_T - K, 0.0)
    hedging_pnl = terminal_portfolio - payoff

    return {
        "times": times,
        "stock_path": stock_path,
        "option_premium": option_premium,
        "deltas": deltas,
        "trades": trades,
        "cash_balances": cash_balances,
        "terminal_portfolio": terminal_portfolio,
        "payoff": payoff,
        "hedging_pnl": hedging_pnl
    }

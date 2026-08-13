import math

import numpy.typing as npt
import numpy as np

from src.derivatives.european_call import (
    black_scholes_call,
    black_scholes_delta,
    european_call_payoff
)

from src.portfolio import run_hedge


def run_delta_hedge(
    stock_paths,
    K,
    sigma,
    T,
    r=0.0,
    transaction_cost_rate=0.0,
    record_history=False
):
    stock_paths = np.asarray(stock_paths, dtype=float)

    S0 = stock_paths[:, 0]

    premium = black_scholes_call(
        S=S0,
        K=K,
        sigma=sigma,
        T=T,
        t=0.0,
        r=r
    )

    def position_fn(S, t):
        return black_scholes_delta(
            S=S,
            K=K,
            sigma=sigma,
            T=T,
            t=t,
            r=r
        )

    def payoff_fn(S_T):
        return european_call_payoff(S_T, K)

    return run_hedge(
        stock_paths=stock_paths,
        initial_premium=premium,
        position_fn=position_fn,
        payoff_fn=payoff_fn,
        T=T,
        r=r,
        transaction_cost_rate=transaction_cost_rate,
        record_history=record_history
    )

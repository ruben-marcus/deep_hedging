import numpy as np
from scipy.special import ndtr


def _scalar_if_scalar(x):
    x = np.asarray(x)

    if x.ndim == 0:
        return x.item()

    return x


def european_call_payoff(S, K):
    S = np.asarray(S, dtype=float)

    payoff = np.maximum(S - K, 0.0)

    return _scalar_if_scalar(payoff)


def black_scholes_call(
    S,
    K,
    sigma,
    T,
    t=0.0,
    r=0.0,
):
    """
    black-scholes price of a european call option

    parameters
    ----------
    S : float
        current stock price

    K : float
        strike price

    sigma : float
        annualized volatility

    T : float
        maturity time in years

    t : float
        current time in years

    r : float
        risk-free rate
    """

    S = np.asarray(S, dtype=float)
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)

    if np.any(S <= 0):
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if np.any(tau < 0):
        raise ValueError("t cannot be greater than T")

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_tau = np.sqrt(tau)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau

        price = S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)

    # at expiry use payoff directly
    price = np.where(
        tau == 0,
        np.maximum(S - K, 0.0),
        price
    )

    return _scalar_if_scalar(price)


def black_scholes_delta(
    S,
    K,
    sigma,
    T,
    t=0.0,
    r=0.0,
) -> float:
    """
    black-scholes delta of a european call option
    """

    S = np.asarray(S, dtype=float)
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)

    if np.any(S <= 0):
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if np.any(tau < 0):
        raise ValueError("t cannot be greater than T")

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2)
              * tau) / (sigma * np.sqrt(tau))

        delta = ndtr(d1)

    # convention at expiration
    expiry_delta = np.where(
        S > K,
        1.0,
        np.where(S < K, 0.0, 0.5)
    )

    delta = np.where(
        tau == 0,
        expiry_delta,
        delta
    )

    return _scalar_if_scalar(delta)

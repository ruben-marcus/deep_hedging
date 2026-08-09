import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    S: float,
    K: float,
    sigma: float,
    T: float,
    t: float = 0.0,
    r: float = 0.0,
) -> float:
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

    if S <= 0:
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    tau = T - t

    if tau < 0:
        raise ValueError("t cannot be greater than T")

    if tau == 0:
        return max(S - K, 0.0)

    sqrt_tau = math.sqrt(tau)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    call_price = S * norm_cdf(d1) - K * math.exp(-r * tau) * norm_cdf(d2)

    return call_price


def black_scholes_delta(
    S: float,
    K: float,
    sigma: float,
    T: float,
    t: float = 0.0,
    r: float = 0.0,
) -> float:
    """
    black-scholes delta of a european call option
    """

    if S <= 0:
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    tau = T - t

    if tau < 0:
        raise ValueError("t cannot be greater than T")

    if tau == 0:
        if S > K:
            return 1.0
        elif S < K:
            return 0.0
        else:
            return 0.5  # convention

    sqrt_tau = math.sqrt(tau)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)

    return norm_cdf(d1)

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatOrArray = float | npt.NDArray[np.float64]


def _scalar_if_scalar(x: npt.ArrayLike) -> FloatOrArray:
    """unwrap 0-d arrays so scalar inputs give scalar outputs"""

    x = np.asarray(x)

    if x.ndim == 0:
        return x.item()

    return x


def _prepare(S: npt.ArrayLike, T: float, t: npt.ArrayLike):
    """common coercion: spot as float array, tau as time to maturity"""

    S = np.asarray(S, dtype=float)
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)

    return S, tau


def _validate(S, K: float, sigma: float, tau) -> None:
    if np.any(S <= 0):
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if np.any(tau < 0):
        raise ValueError("t cannot be greater than T")


def _d1_d2(S, K: float, sigma: float, tau, r: float):
    """tau == 0 gives inf/nan, callers overwrite those entries"""

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_tau = np.sqrt(tau)

        d1 = (
            np.log(S / K)
            + (r + 0.5 * sigma**2) * tau
        ) / (sigma * sqrt_tau)

        d2 = d1 - sigma * sqrt_tau

    return d1, d2


def european_call_payoff(S: npt.ArrayLike, K: float) -> FloatOrArray:
    """terminal payoff max(S - K, 0) of a european call"""

    S = np.asarray(S, dtype=float)

    payoff = np.maximum(S - K, 0.0)

    return _scalar_if_scalar(payoff)


def black_scholes_call(
    S: npt.ArrayLike,
    K: float,
    sigma: float,
    T: float,
    t: npt.ArrayLike = 0.0,
    r: float = 0.0,
) -> FloatOrArray:
    """
    black-scholes price of a european call option

    parameters
    ----------
    S : float or array
        current stock price

    K : float
        strike price

    sigma : float
        annualized volatility

    T : float
        maturity time in years

    t : float or array
        current time in years

    r : float
        risk-free rate

    returns
    -------
    float or array
        call price, same shape as S
    """

    S, tau = _prepare(S, T, t)
    _validate(S, K, sigma, tau)
    d1, d2 = _d1_d2(S, K, sigma, tau, r)

    price = S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)

    # at expiry use payoff directly
    price = np.where(
        tau == 0,
        np.maximum(S - K, 0.0),
        price,
    )

    return _scalar_if_scalar(price)


def black_scholes_delta(
    S: npt.ArrayLike,
    K: float,
    sigma: float,
    T: float,
    t: npt.ArrayLike = 0.0,
    r: float = 0.0,
) -> FloatOrArray:
    """
    black-scholes delta of a european call option

    same parameters as black_scholes_call
    hedge ratio used as benchmark policy: shares to hold per short call
    """

    S, tau = _prepare(S, T, t)
    _validate(S, K, sigma, tau)
    d1, d2 = _d1_d2(S, K, sigma, tau, r)

    delta = ndtr(d1)

    # convention at expiration
    expiry_delta = np.where(
        S > K,
        1.0,
        np.where(S < K, 0.0, 0.5),
    )

    delta = np.where(
        tau == 0,
        expiry_delta,
        delta,
    )

    return _scalar_if_scalar(delta)

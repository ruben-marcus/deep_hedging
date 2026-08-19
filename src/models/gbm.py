import numpy as np

from models.base import SimulationResult


def simulate_gbm(
    S0: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    r: float = 0.0,
    seed: int | None = None,
) -> SimulationResult:
    """
    simulate geometric brownian motion under risk-neutral measure with r = 0

    parameters
    ----------
    S0 : float
        initial stock price

    sigma : float
        annualized volatility

    T : float
        time horizon in years

    n_steps : int
        number of simulation/hedging steps

    n_paths : int
        number of monte carlo paths

    r : float
        risk-free rate

    seed : int or None
        random seed for reproducibility

    returns
    -------
    SimulationResult
        spot of shape (n_paths, n_steps + 1)
        variance is None

    """

    if S0 <= 0:
        raise ValueError("S0 must be positive")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    if T <= 0:
        raise ValueError("T must be positive")

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    dt = T / n_steps

    rng = np.random.default_rng(seed)

    # one indepenedent N(0, 1) shock for each path and timestamp
    Z = rng.standard_normal(size=(n_paths, n_steps))

    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)

    return SimulationResult(spot=paths, variance=None)

import numpy as np

from src.models.base import SimulationResult


def simulate_heston(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    r: float = 0.0,
    seed: int | None = None
) -> SimulationResult:
    """
    simulate heston using full-truncation euler for variance and log-euler for stocks

    parameters
    ----------
    S0 : float
        initial stock price

    v0 : float
        initial instantaneous variance (volatility**2)

    kappa : float
        mean-reversion speed of variance

    theta : float
        long-run average variance

    xi : float
        volatility of variance (vol-of-vol)

    rho : float
        correlation between spot and variance shocks
        (almost always negative)

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
        spot and non-negative variance, both (n_paths, n_steps + 1)
    """

    if S0 <= 0:
        raise ValueError("S0 must be positive")

    if v0 < 0:
        raise ValueError("v0 must be non-negative")

    if kappa < 0:
        raise ValueError("kappa must be non-negative")

    if theta < 0:
        raise ValueError("theta must be non-negative")

    if xi < 0:
        raise ValueError("xi must be non-negative")

    if not -1 <= rho <= 1:
        raise ValueError("rho must lie in [-1, 1]")

    if T <= 0:
        raise ValueError("T must be positive")

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)

    spot = np.empty((n_paths, n_steps + 1), dtype=float)
    variance = np.empty((n_paths, n_steps + 1), dtype=float)

    spot[:, 0] = S0
    variance[:, 0] = v0

    # internal raw variance state for full truncation
    raw_v = np.full(n_paths, v0, dtype=float)

    for i in range(n_steps):
        Z_v = rng.standard_normal(n_paths)
        Z_independent = rng.standard_normal(n_paths)

        Z_s = rho * Z_v + np.sqrt(1.0 - rho**2) * Z_independent

        v_positive = np.maximum(raw_v, 0.0)

        # spot evolution
        spot[:, i + 1] = spot[:, i] * np.exp(
            (r - 0.5 * v_positive) * dt + np.sqrt(v_positive) * sqrt_dt * Z_s
        )

        # full-truncation euler variance update
        raw_v = raw_v + kappa * (theta - v_positive) * dt + \
            xi * np.sqrt(v_positive) * sqrt_dt * Z_v

        # store economically meaningful non-negative variance
        variance[:, i + 1] = np.maximum(raw_v, 0.0)

    return SimulationResult(spot=spot, variance=variance)

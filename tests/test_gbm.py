import numpy as np
import pytest

from models.gbm import simulate_gbm

from conftest import N_SIGMA, standard_error


def test_shape_and_values(gbm_paths, market):
    assert gbm_paths.spot.shape == (20_000, 31)
    assert gbm_paths.n_paths == 20_000
    assert gbm_paths.n_steps == 30

    assert np.all(gbm_paths.spot[:, 0] == market["S0"])
    assert np.all(gbm_paths.spot > 0)


def test_variance_is_constant_sigma_squared(gbm_paths, market):
    """gbm has constant instantaneous variance"""

    assert gbm_paths.variance is not None
    assert gbm_paths.variance.shape == gbm_paths.spot.shape
    assert np.all(gbm_paths.variance == market["sigma"] ** 2)


def test_seed_reproducibility(market):
    kwargs = dict(
        S0=market["S0"],
        sigma=market["sigma"],
        T=market["T"],
        n_steps=10,
        n_paths=500,
    )

    a = simulate_gbm(**kwargs, seed=1).spot
    b = simulate_gbm(**kwargs, seed=1).spot
    c = simulate_gbm(**kwargs, seed=2).spot

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_log_return_moments(market):
    """log(S_T / S0) ~ Normal((r - sigma^2 / 2)T, sigma^2 T)"""

    r, sigma, T = 0.03, market["sigma"], market["T"]

    sim = simulate_gbm(
        S0=market["S0"],
        sigma=sigma,
        T=T,
        n_steps=30,
        n_paths=400_000,
        r=r,
        seed=3,
    )

    log_returns = np.log(sim.spot[:, -1] / market["S0"])

    assert log_returns.mean() == pytest.approx(
        (r - 0.5 * sigma**2) * T,
        abs=N_SIGMA * standard_error(log_returns),
    )

    assert log_returns.var(ddof=1) == pytest.approx(
        sigma**2 * T,
        rel=0.02,
    )


@pytest.mark.parametrize("kwargs, message", [
    (dict(
        S0=0.0,
        sigma=0.20,
        T=1.0,
        n_steps=10,
        n_paths=10,
    ), "S0 must be positive"),

    (dict(
        S0=100.0,
        sigma=-0.20,
        T=1.0,
        n_steps=10,
        n_paths=10,
    ), "sigma must be non-negative"),

    (dict(
        S0=100.0,
        sigma=0.20,
        T=0.0,
        n_steps=10,
        n_paths=10,
    ), "T must be positive"),

    (dict(
        S0=100.0,
        sigma=0.20,
        T=1.0,
        n_steps=0,
        n_paths=10,
    ), "n_steps must be positive"),

    (dict(
        S0=100.0,
        sigma=0.20,
        T=1.0,
        n_steps=10,
        n_paths=0,
    ), "n_paths must be positive"),
])
def test_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_gbm(**kwargs)

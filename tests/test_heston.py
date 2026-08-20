import numpy as np
import pytest

from models.heston import simulate_heston

from conftest import N_SIGMA, standard_error


def test_shape_and_values(heston_paths):
    assert heston_paths.spot.shape == (20_000, 31)
    assert heston_paths.variance.shape == heston_paths.spot.shape

    assert np.all(heston_paths.spot > 0)
    assert np.all(heston_paths.variance >= 0)


def test_seed_reproducibility(heston_params):
    kwargs = dict(**heston_params, n_steps=10, n_paths=500)

    a = simulate_heston(**kwargs, seed=6)
    b = simulate_heston(**kwargs, seed=6)
    c = simulate_heston(**kwargs, seed=7)

    assert np.array_equal(a.spot, b.spot)
    assert np.array_equal(a.variance, b.variance)
    assert not np.array_equal(a.spot, c.spot)


def test_terminal_mean_is_forward(heston_params):
    """spot is martingale under r = 0, so E[S_T] = S0"""

    sim = simulate_heston(
        **heston_params,
        n_steps=30,
        n_paths=400_000,
        seed=8,
    )

    terminal = sim.spot[:, -1]
    expected = heston_params["S0"] * np.exp(
        heston_params["r"] * heston_params["T"]
    )

    assert terminal.mean() == pytest.approx(
        expected, abs=N_SIGMA * standard_error(terminal)
    )


def test_zero_vol_of_vol_freezes_variance(heston_params):
    """xi = 0 and v0 = theta: heston -> gbm with sigma = sqrt(theta)"""

    params = {
        **heston_params,
        "xi": 0.0,
        "v0": heston_params["theta"],
    }

    sim = simulate_heston(
        **params,
        n_steps=30,
        n_paths=500_000,
        seed=9,
    )

    assert np.all(sim.variance == heston_params["theta"])

    log_returns = np.log(sim.spot[:, -1] / params["S0"])
    expected_variance = heston_params["theta"] * heston_params["T"]

    assert log_returns.var(ddof=1) == pytest.approx(
        expected_variance, rel=0.02
    )


@pytest.mark.parametrize("override, message", [
    (dict(S0=0.0), "S0 must be positive"),
    (dict(v0=-0.04), "v0 must be non-negative"),
    (dict(kappa=-2.0), "kappa must be non-negative"),
    (dict(theta=-0.04), "theta must be non-negative"),
    (dict(xi=-0.30), "xi must be non-negative"),
    (dict(rho=1.5), r"rho must lie in \[-1, 1\]"),
    (dict(T=0.0), "T must be positive"),
])
def test_validation(heston_params, override, message):
    params = {
        **heston_params,
        **override,
        "n_steps": 5,
        "n_paths": 5,
    }

    with pytest.raises(ValueError, match=message):
        simulate_heston(**params)

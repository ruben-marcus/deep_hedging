import numpy as np
import pytest

from derivatives.european_call import (
    black_scholes_call,
    black_scholes_delta,
    european_call_payoff,
)
from models.gbm import simulate_gbm

from conftest import N_SIGMA, standard_error


def test_reference_value(market):
    """regression pin: atm, 20% vol, 1y, zero rates"""

    price = black_scholes_call(
        S=market["S0"],
        K=market["K"],
        sigma=market["sigma"],
        T=market["T"],
        r=market["r"],
    )

    assert price == pytest.approx(7.965567, abs=1e-4)


def test_monte_carlo_matches_closed_form(market):
    """cross-validate simulator against pricer"""

    n_paths = 400_000

    sim = simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=market["T"],
        n_steps=1,
        n_paths=n_paths,
        r=market["r"],
        seed=5,
    )

    payoffs = european_call_payoff(sim.spot[:, -1], market["K"])
    discounted = np.exp(-market["r"] * market["T"]) * payoffs

    analytic = black_scholes_call(
        S=market["S0"],
        K=market["K"],
        sigma=market["sigma"],
        T=market["T"],
        r=market["r"],
    )

    assert discounted.mean() == pytest.approx(
        analytic, abs=N_SIGMA * standard_error(discounted)
    )


def test_delta_is_derivative_of_price(market):
    """delta == d(price)/dS"""

    h = 1e-4

    for S in (70.0, 95.0, 100.0, 105.0, 130.0):
        up = black_scholes_call(
            S=S + h,
            K=market["K"],
            sigma=market["sigma"],
            T=market["T"],
            r=market["r"],
        )

        down = black_scholes_call(
            S=S - h,
            K=market["K"],
            sigma=market["sigma"],
            T=market["T"],
            r=market["r"],
        )

        finite_difference = (up - down) / (2 * h)

        analytic = black_scholes_delta(
            S=S,
            K=market["K"],
            sigma=market["sigma"],
            T=market["T"],
            r=market["r"],
        )

        assert finite_difference == pytest.approx(analytic, rel=1e-6)


@pytest.mark.parametrize("kwargs, message", [
    (dict(
        S=-100.0,
        K=100.0,
        sigma=0.20,
        T=1.0,
    ), "S must be positive"),

    (dict(
        S=100.0,
        K=-100.0,
        sigma=0.20,
        T=1.0,
    ), "K must be positive"),

    (dict(
        S=100.0,
        K=100.0,
        sigma=0.0,
        T=1.0,
    ), "sigma must be positive"),

    (dict(
        S=100.0,
        K=100.0,
        sigma=0.20,
        T=1.0,
        t=2.0,
    ), "t cannot be greater than T"),

])
def test_validation(kwargs, message):
    for fn in (black_scholes_call, black_scholes_delta):
        with pytest.raises(ValueError, match=message):
            fn(**kwargs)


def test_payoff():
    S = [80.0, 100.0, 120.0]

    assert np.allclose(
        european_call_payoff(S, 100.0),
        [0.0, 0.0, 20.0]
    )

    assert isinstance(european_call_payoff(120.0, 100.0), float)

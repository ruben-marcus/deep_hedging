import math

import numpy as np
import pytest
import torch
from torch import nn

from models.gbm import simulate_gbm
from models.heston import simulate_heston

# compare simulated results to analytic values with tolerance of
# N_SIGMA standard errors
N_SIGMA = 4.0

SEED = 123456789


@pytest.fixture(scope="session")
def market():
    """option and market parameters shared by most tests"""

    return dict(
        S0=100.0,
        K=100.0,
        sigma=0.20,
        T=1.0,
        r=0.0,
    )


@pytest.fixture(scope="session")
def heston_params():
    """xi > 0 and rho < 0"""

    return dict(
        S0=100.0,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.30,
        rho=-0.7,
        T=1.0,
        r=0.0,
    )


@pytest.fixture(scope="session")
def gbm_paths(market):
    return simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=market["T"],
        n_steps=30,
        n_paths=20_000,
        r=market["r"],
        seed=SEED,
    )


@pytest.fixture(scope="session")
def heston_paths(heston_params):
    return simulate_heston(
        **heston_params,
        n_steps=30,
        n_paths=20_000,
        seed=SEED,
    )


def standard_error(samples) -> float:
    """standard error of mean, for monte carlo tolerance"""

    samples = np.asarray(samples, dtype=float)

    return float(samples.std(ddof=1) / np.sqrt(samples.size))


class ConstantPolicy(nn.Module):
    """holds fixed number of shares regardless of state"""

    def __init__(self, shares: float = 0.5):
        super().__init__()
        self.shares = shares

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state[:, 0], self.shares)


class BlackScholesPolicy(nn.Module):
    """black-scholes delta, computed from build_state state"""

    def __init__(
        self,
        K: float,
        T: float,
        sigma: float,
        r: float = 0.0,
    ):
        super().__init__()
        self.K, self.T, self.sigma, self.r = K, T, sigma, r

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        log_moneyness = state[:, 0]
        tau = state[:, 2] * self.T

        S = self.K * torch.exp(log_moneyness)

        safe_tau = torch.clamp(tau, min=1e-12)

        d1 = (
            torch.log(S / self.K)
            + (self.r + 0.5 * self.sigma**2) * safe_tau
        ) / (self.sigma * torch.sqrt(safe_tau))

        delta = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))

        expiry = (S > self.K).to(S.dtype)

        return torch.where(tau > 0, delta, expiry)


@pytest.fixture
def constant_policy():
    return ConstantPolicy(0.5)


@pytest.fixture
def bs_policy(market):
    return BlackScholesPolicy(
        K=market["K"],
        T=market["T"],
        sigma=market["sigma"],
        r=market["r"],
    )

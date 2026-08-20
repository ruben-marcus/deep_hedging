import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gbm import simulate_gbm
from portfolio import run_hedge_torch
from experiment import get_device, set_seeds
from losses import CVaRLoss, MSEHedgingLoss
from strategies.neural_hedge import NeuralHedge
from train import collect_pnl, evaluate_loss, fit


@pytest.fixture
def dataset(market):
    sim = simulate_gbm(
        S0=market["S0"],
        sigma=market["sigma"],
        T=1.0,
        n_steps=10,
        n_paths=200,
        seed=19,
    )

    spot = torch.tensor(sim.spot, dtype=torch.float32)
    variance = torch.tensor(np.asarray(sim.variance), dtype=torch.float32)

    return TensorDataset(spot, variance)


@pytest.fixture
def loader(dataset):
    return DataLoader(dataset, batch_size=50, shuffle=False)


@pytest.fixture
def run_batch(market):
    def _run_batch(model, batch):
        spot, variance = batch

        return run_hedge_torch(
            model=model,
            spot_paths=spot,
            variance_paths=variance,
            K=market["K"],
            T=1.0,
            premium=8.0,
            transaction_cost_rate=0.001,
        )["pnl"]

    return _run_batch


def test_fit_reduces_loss(loader, run_batch):
    torch.manual_seed(0)
    model = NeuralHedge()

    history = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_fn=MSEHedgingLoss(),
        run_batch=run_batch,
        n_epochs=5,
    )

    assert history[-1]["train_loss"] < history[0]["train_loss"]


class _SingleWeight(torch.nn.Module):
    """minimal policy with one parameter for exercising training loop"""

    def __init__(self, start: float = 0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([start]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.w.expand(state.shape[0])


def _quadratic_run_batch(model, batch):
    """mock batch runner mapping output to (w - 2)"""

    (x,) = batch
    return model(x) - 2.0


def test_fit_restores_best_weights():
    """
    verify that fit() restores model weights from epoch with lowest val loss

    gradient descent on (w - 2)^2 with lr = 1.1 overshoots,
    so val loss is best at first epoch
    """

    dataset = TensorDataset(torch.zeros(10, 1))
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    model = _SingleWeight(start=0.0)
    loss_fn = MSEHedgingLoss()

    history = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=1.1),
        loss_fn=loss_fn,
        run_batch=_quadratic_run_batch,
        n_epochs=5,
    )

    val_losses = [h["val_loss"] for h in history]

    assert val_losses == sorted(val_losses), "expected steadily worse epochs"
    assert min(val_losses) == val_losses[0]

    # w after first step: 0 - 1.1 * 2 * (0 - 2) = 4.4
    assert float(model.w.detach()) == pytest.approx(4.4, rel=1e-5)

    restored = evaluate_loss(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        run_batch=_quadratic_run_batch,
    )

    assert restored == pytest.approx(min(val_losses), rel=1e-6)
    assert restored < val_losses[-1]


def test_fit_reports_divergence():
    """raise runtime error if training diverges to nan"""

    dataset = TensorDataset(torch.zeros(10, 1))
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    model = _SingleWeight(start=float("nan"))

    with pytest.raises(RuntimeError, match="diverged|no .*validation loss"):
        fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=MSEHedgingLoss(),
            run_batch=_quadratic_run_batch,
            n_epochs=5
        )


def test_fit_restores_loss_parameters(loader, run_batch):
    """e.g. trainable eta in CVaRLoss"""

    torch.manual_seed(20)
    model = NeuralHedge()
    loss_fn = CVaRLoss(alpha=0.95)

    parameters = list(model.parameters()) + list(loss_fn.parameters())

    fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.Adam(parameters, lr=1e-2),
        loss_fn=loss_fn,
        run_batch=run_batch,
        n_epochs=5,
    )

    eta = loss_fn.eta.detach()

    assert torch.isfinite(eta)
    assert float(eta) != 0.0


def test_collect_pnl_shape_and_order(dataset, run_batch):
    torch.manual_seed(21)
    model = NeuralHedge()

    loader = DataLoader(dataset, batch_size=50, shuffle=False)
    collected = collect_pnl(model, loader, run_batch)

    assert collected.shape == (len(dataset),)

    full = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    direct = collect_pnl(model, full, run_batch)

    assert np.allclose(collected, direct, atol=1e-6)


def test_seed_reproducibility():
    set_seeds(22)
    a = (np.random.rand(5), torch.rand(5))

    set_seeds(22)
    b = (np.random.rand(5), torch.rand(5))

    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_get_device_is_usable():
    device = get_device()

    assert isinstance(device, torch.device)
    assert torch.zeros(1, device=device).device.type == device.type

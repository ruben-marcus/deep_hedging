import copy
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# run_batch(model, batch) -> pnl of shape (batch_size,)
RunBatch = Callable[[nn.Module, Any], torch.Tensor]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    run_batch: RunBatch
) -> float:
    """one pass over loader, returns sample-weighted mean loss"""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad()

        pnl = run_batch(model, batch)
        loss = loss_fn(pnl)
        loss.backward()
        optimizer.step()

        batch_size = pnl.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    run_batch: RunBatch
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        pnl = run_batch(model, batch)
        loss = loss_fn(pnl)

        batch_size = pnl.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    run_batch: RunBatch,
    n_epochs: int
) -> list[dict]:
    """
    train for n_epochs and restore weights with the best validation loss

    pass loss parameters (e.g. eta of CVaRLoss) to optimizer as well

    returns
    -------
    list of dicts, one per epoch, with epoch, train_loss, and val_loss
    """

    history = []

    best_val_loss = float("inf")
    best_model_state = None
    best_loss_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            run_batch=run_batch
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            run_batch=run_batch
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            if isinstance(loss_fn, torch.nn.Module):
                best_loss_state = copy.deepcopy(loss_fn.state_dict())

    model.load_state_dict(best_model_state)

    if (
        best_loss_state is not None
        and isinstance(loss_fn, torch.nn.Module)
    ):
        loss_fn.load_state_dict(best_loss_state)

    return history


@torch.no_grad()
def collect_pnl(
    model: nn.Module,
    loader: DataLoader,
    run_batch: RunBatch
) -> np.ndarray:
    """pnl over every path in loader, shape (n_paths,)"""

    model.eval()

    pnl_batches = []

    for batch in loader:
        pnl = run_batch(model, batch)
        pnl_batches.append(pnl.cpu())

    return torch.cat(pnl_batches).numpy()
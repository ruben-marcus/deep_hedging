import copy
import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    run_batch
):
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
    model,
    loader,
    loss_fn,
    run_batch
):
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
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    run_batch,
    n_epochs
):
    history = []

    best_val_loss = float("inf")
    best_state = None

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
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return history


@torch.no_grad()
def collect_pnl(model, loader, run_batch):
    model.eval()

    pnl_batches = []

    for batch in loader:
        pnl = run_batch(model, batch)
        pnl_batches.append(pnl.cpu())

    return torch.cat(pnl_batches).numpy()

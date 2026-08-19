import torch


def mse_hedging_loss(pnl: torch.Tensor) -> torch.Tensor:
    """
    functional form of MSEHedgingLoss, kept for notebooks 05-07

    for new stuff use classes in src.losses
    """

    return torch.mean(pnl**2)
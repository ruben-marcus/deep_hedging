import torch


def mse_hedging_loss(pnl):
    return torch.mean(pnl**2)

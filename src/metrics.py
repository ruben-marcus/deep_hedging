from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch


def _as_numpy(x: npt.ArrayLike | torch.Tensor) -> np.ndarray:
    """accept numpy or torch, gives back a flat float array"""

    if torch.is_tensor(x):
        x = x.detach().cpu()

    return np.asarray(x, dtype=float).ravel()


def empirical_cvar(
    pnl: npt.ArrayLike | torch.Tensor,
    alpha: float = 0.95,
) -> float:
    """mean loss in the worst (1 - alpha) fraction of outcomes"""

    pnl = _as_numpy(pnl)
    n = pnl.size

    if n == 0:
        raise ValueError("pnl must be non-empty")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    losses = -pnl

    tail_count = max(1, int(np.ceil((1.0 - alpha) * n)))

    # partition puts the tail_count largest losses last
    worst_losses = np.partition(losses, n - tail_count)[n - tail_count:]

    return float(worst_losses.mean())


def pnl_metrics(
    pnl: npt.ArrayLike | torch.Tensor,
    cvar_alphas: Sequence[float] = (0.95, 0.99),
) -> dict[str, float]:
    """
    distribution summary of terminal hedging pnl

    returns
    -------
    dict with mean_pnl, std_pnl, rmse, q05, q01
    and one cvar{alpha}_loss entry per alpha in cvar_alphas
    """

    pnl = _as_numpy(pnl)

    metrics = {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "rmse": float(np.sqrt(np.mean(pnl**2))),
        "q05": float(np.quantile(pnl, 0.05)),
        "q01": float(np.quantile(pnl, 0.01)),
    }

    for alpha in cvar_alphas:
        key = f"cvar{int(round(alpha * 100))}_loss"
        metrics[key] = empirical_cvar(pnl, alpha=alpha)

    return metrics


def hedge_metrics(
    result: Mapping[str, npt.ArrayLike | torch.Tensor],
    cvar_alphas: Sequence[float] = (0.95, 0.99),
) -> dict[str, float]:
    """
    pnl_metrics + average cost and turnover

    takes result dict from run_hedge or run_hedge_torch
    """

    metrics = pnl_metrics(result["pnl"], cvar_alphas=cvar_alphas)

    metrics["mean_tc"] = float(
        np.mean(_as_numpy(result["transaction_costs"]))
    )
    metrics["mean_share_turnover"] = float(
        np.mean(_as_numpy(result["share_turnover"]))
    )

    if "notional_turnover" in result:
        metrics["mean_notional_turnover"] = float(
            np.mean(_as_numpy(result["notional_turnover"]))
        )

    return metrics

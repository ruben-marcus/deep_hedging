import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    output of a path simulator

    spot : np.ndarray
        shape (n_paths, n_steps + 1)

    variance : np.ndarray or None
        instantaneous variance, same shape as spot
        None for models with constant volatility (e.g. gbm)
    """

    spot: np.ndarray
    variance: np.ndarray | None = None

    def __post_init__(self):
        if self.variance is not None and self.variance.shape != self.spot.shape:
            raise ValueError("variance and spot must have the same shape")

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1

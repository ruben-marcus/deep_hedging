import torch
import torch.nn as nn


class NeuralHedge(nn.Module):
    """
    feedforward hedging policy, state -> share position

    state built by run_hedge_torch:
        log moneyness, scaled variance, time to maturity, current shares

    output is unconstrained, may take any long or short position
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        return self.network(state).squeeze(-1)
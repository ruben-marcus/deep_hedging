import torch
import torch.nn as nn


class NeuralHedge(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def get_device() -> torch.device:
    """cuda, then apple mps, then cpu"""

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def set_seeds(seed: int) -> None:
    """seed numpy and torch together"""

    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """train is shuffled from seeded generator, val and test are not"""

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training datasets

"""Where a Studio training run gets its data.

One owner for the loaders, separate from the job that supervises a run. Every
dataset here is named in the training contract's supported set, so a request
can only ask for one that exists.
"""

from __future__ import annotations

from typing import Any


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch before building data or model state.

    Parameters
    ----------
    seed : int
        Seed from the validated training configuration.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in this environment
        torch.cuda.manual_seed_all(seed)


def _make_synthetic(batch_size: int, *, rates: bool = False) -> tuple[Any, Any, int, int]:
    """Generate synthetic classification data; ``rates`` maps the draws into ``(0, 1)``.

    The conversion route encodes inputs as rates, so it takes the same draws
    through the logistic function. Torch is imported inside the loaders because
    it is an optional extra.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 512
    n_inputs = 64
    n_classes = 10
    features = torch.randn(n_samples, n_inputs)
    features = torch.sigmoid(features) if rates else features
    labels = torch.randint(0, n_classes, (n_samples,))
    split = int(0.8 * n_samples)
    train_dataset = TensorDataset(features[:split], labels[:split])
    test_dataset = TensorDataset(features[split:], labels[split:])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_dataset, batch_size=batch_size, drop_last=True),
        n_inputs,
        n_classes,
    )


def _load_mnist(batch_size: int, *, rates: bool = False) -> tuple[Any, Any, int, int]:
    """Load MNIST through torchvision; ``rates`` keeps pixels in ``[0, 1]`` unnormalised.

    Raises
    ------
    RuntimeError
        torchvision is not installed. No other data is substituted: a run that
        asked for MNIST and trained on something else would carry the name of
        a dataset it never saw.
    """
    from torch.utils.data import DataLoader

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "The MNIST dataset needs torchvision, which is not installed here; "
            "install it or choose the synthetic dataset. No other data was substituted."
        ) from exc
    normalise = [] if rates else [transforms.Normalize((0.1307,), (0.3081,))]
    transform = transforms.Compose([transforms.ToTensor(), *normalise])
    train_dataset = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("~/.cache/mnist", train=False, transform=transform)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size),
        784,
        10,
    )

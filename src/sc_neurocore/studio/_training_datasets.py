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


def _make_synthetic(batch_size: int) -> tuple[Any, Any, int, int]:
    """Generate synthetic classification data for quick demonstrations.

    Torch is imported inside the loaders because it is an optional extra: a
    Studio without it must still import this module to say which datasets exist.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 512
    n_inputs = 64
    n_classes = 10
    features = torch.randn(n_samples, n_inputs)
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


def _load_mnist(batch_size: int) -> tuple[Any, Any, int, int]:
    """Load MNIST through torchvision, or use the synthetic fallback."""
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            "~/.cache/mnist",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            "~/.cache/mnist",
            train=False,
            transform=transform,
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size),
            784,
            10,
        )
    except ImportError:
        return _make_synthetic(batch_size)

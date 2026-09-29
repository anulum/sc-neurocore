# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event resume saved-state comparison

"""Compare complete states downloaded from real training artifact routes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def same_state(left: object, right: object) -> bool:
    """Compare all tensors and optimiser/RNG primitives without ZIP framing.

    Parameters
    ----------
    left, right:
        Saved states loaded with Torch's restricted weights-only reader.

    Returns
    -------
    bool
        Whether every type, dictionary field, sequence element and tensor agrees.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return bool(left == right)


def main() -> None:
    """Refuse any semantic difference between two actual saved training states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("uninterrupted", type=Path)
    args = parser.parse_args()
    resumed: object = torch.load(args.resumed, weights_only=True, map_location="cpu")
    uninterrupted: object = torch.load(args.uninterrupted, weights_only=True, map_location="cpu")
    if not same_state(resumed, uninterrupted):
        raise ValueError(
            "resumed model, optimiser or RNG state differs from uninterrupted training"
        )
    print("Complete saved training states match")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning backend factory

"""Backend selection without coupling Rust construction to PyTorch imports."""

from __future__ import annotations

from typing import Any

from .learning_rust_layer import RustRuleLayer
from .learning_validation import RULE_STDP
from .learning_wgpu import RustWgpuRuleLayer


def create_plasticity_layer(
    count: int,
    rule_type: int = RULE_STDP,
    backend: str = "torch",
    autograd: bool = True,
    **kwargs: Any,
) -> Any:
    """Construct a Torch, Rust-Rayon, or Rust-WGPU plasticity layer.

    Rust backends remain available when PyTorch is not installed.  Torch is
    imported only when explicitly selected so optional dependency failures do
    not disable native learning.
    """
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    selected = backend.casefold()
    if selected == "rust":
        return RustRuleLayer(count=count, rule_type=rule_type, **kwargs)
    if selected == "rust-wgpu":
        return RustWgpuRuleLayer(count=count, rule_type=rule_type, **kwargs)
    if selected == "torch":
        try:
            from .learning_torch import TorchRuleLayer
        except ImportError as exc:
            raise ImportError(
                "the Torch learning backend requires the 'torch' extra; "
                "install sc-neurocore[torch] or select 'rust'/'rust-wgpu'"
            ) from exc
        return TorchRuleLayer(
            count=count,
            rule_type=rule_type,
            autograd=autograd,
            **kwargs,
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'torch', 'rust', or 'rust-wgpu'")

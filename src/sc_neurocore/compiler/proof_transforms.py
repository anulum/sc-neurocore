# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Opt-in formal-proof transform registry

"""Dispatch opt-in RTL transforms used by formal proof flows.

The compiler's ordinary Verilog generation path must stay behavioural: it emits
the concrete RTL requested by the caller. Unbounded equivalence proofs sometimes
need a proof-only RTL surface, such as state observation taps or lifted
multiplier products. This module is the explicit dispatch boundary for those
proof transforms. The transforms are discoverable and selectable by proof
pipelines, but none is enabled by default in production compilation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .operator_abstraction import LiftedSignal, abstract_to_free_inputs
from .whitebox_taps import StateTap, expose_state_taps

ProofTransformKind: TypeAlias = Literal["whitebox_taps", "operator_abstraction"]

__all__ = [
    "LiftedSignal",
    "PROOF_TRANSFORMS",
    "ProofTransform",
    "ProofTransformKind",
    "StateTap",
    "abstract_to_free_inputs",
    "apply_proof_transform",
    "expose_state_taps",
    "get_proof_transform",
    "list_proof_transforms",
]


@dataclass(frozen=True)
class ProofTransform:
    """Metadata for one selectable formal-proof RTL transform.

    Attributes
    ----------
    kind : ProofTransformKind
        Stable dispatch key accepted by :func:`apply_proof_transform`.
    module : str
        Import path that owns the concrete transform implementation.
    entrypoint : str
        Public callable name inside ``module``.
    purpose : str
        Human-readable reason the proof pipeline may select the transform.
    default_enabled : bool
        Whether ordinary compiler emission enables the transform by default.
        Proof transforms remain opt-in, so current registry entries are false.
    """

    kind: ProofTransformKind
    module: str
    entrypoint: str
    purpose: str
    default_enabled: bool = False


PROOF_TRANSFORMS: tuple[ProofTransform, ...] = (
    ProofTransform(
        kind="whitebox_taps",
        module="sc_neurocore.compiler.whitebox_taps",
        entrypoint="expose_state_taps",
        purpose="Expose internal state as observation outputs for unbounded k-induction.",
    ),
    ProofTransform(
        kind="operator_abstraction",
        module="sc_neurocore.compiler.operator_abstraction",
        entrypoint="abstract_to_free_inputs",
        purpose="Lift selected combinational products to shared free inputs for tractable proofs.",
    ),
)


def list_proof_transforms() -> tuple[ProofTransform, ...]:
    """Return the formal-proof transforms available to opt-in proof pipelines.

    Returns
    -------
    tuple[ProofTransform, ...]
        Immutable registry entries. Every entry is disabled by default for
        ordinary compilation and must be selected explicitly by proof tooling.
    """
    return PROOF_TRANSFORMS


def get_proof_transform(kind: str) -> ProofTransform:
    """Return the registry entry for ``kind``.

    Parameters
    ----------
    kind : str
        Stable transform key, such as ``"whitebox_taps"`` or
        ``"operator_abstraction"``.

    Returns
    -------
    ProofTransform
        Registry metadata for the requested transform.

    Raises
    ------
    KeyError
        If ``kind`` is not registered.
    """
    for transform in PROOF_TRANSFORMS:
        if transform.kind == kind:
            return transform
    valid = ", ".join(transform.kind for transform in PROOF_TRANSFORMS)
    raise KeyError(f"unknown proof transform {kind!r}; expected one of: {valid}")


def apply_proof_transform(
    kind: ProofTransformKind,
    verilog: str,
    *,
    top: str,
    taps: Sequence[StateTap] | None = None,
    signals: Sequence[LiftedSignal] | None = None,
) -> str:
    """Apply one registered proof-only RTL transform.

    Parameters
    ----------
    kind : ProofTransformKind
        Transform dispatch key.
    verilog : str
        Verilog source containing ``top``.
    top : str
        Module name passed to the selected transform.
    taps : sequence of StateTap, optional
        State taps required when ``kind`` is ``"whitebox_taps"``.
    signals : sequence of LiftedSignal, optional
        Lifted signals required when ``kind`` is ``"operator_abstraction"``.

    Returns
    -------
    str
        Transformed Verilog source.

    Raises
    ------
    KeyError
        If ``kind`` is not registered.
    ValueError
        If the selected transform's required payload is missing.
    """
    if kind == "whitebox_taps":
        if taps is None:
            raise ValueError("whitebox_taps requires taps")
        return expose_state_taps(verilog, top=top, taps=list(taps))

    if kind == "operator_abstraction":
        if signals is None:
            raise ValueError("operator_abstraction requires signals")
        return abstract_to_free_inputs(verilog, top=top, signals=list(signals))

    valid = ", ".join(transform.kind for transform in PROOF_TRANSFORMS)
    raise KeyError(f"unknown proof transform {kind!r}; expected one of: {valid}")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dual-axis catalogue-to-silicon tier scoring

"""Dual-axis readiness scoring for a model descriptor.

The catalogue-to-silicon master plan measures every model on two orthogonal
axes, so the programme's progress is machine-countable and Studio can surface
honest per-model readiness:

* **Science axis (S0-S5)** — from "identifies a real model" up to "validated by
  the metric appropriate to its class, with committed evidence". S0-S3 reuse the
  curation kernel :func:`~sc_neurocore.neurons.model_descriptor.descriptor_completeness_tier`;
  S4 adds faithful discretised dynamics; S5 adds class-correct validation.
* **Silicon axis (H0-H5)** — from "lowers to compile-clean RTL" up to "tool-level
  signed PPA". ``None`` means the model has no silicon evidence yet.

Every tier is a pure function of the descriptor's recorded evidence facets
(:class:`~sc_neurocore.neurons.model_descriptor.Validation` and
:class:`~sc_neurocore.neurons.model_descriptor.Silicon`): a rung is credited only
when its proof anchor is present, so a tier can never be inflated ahead of the
committed evidence (master plan invariant I7 — science authored, engineering
derived). Nothing here mutates or stores a tier; the scores are computed on read.
"""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.neurons.model_descriptor import (
    ModelDescriptor,
    descriptor_completeness_tier,
)

# The silicon axis, ordered from compile-clean RTL to signed PPA. The index of a
# label is its numeric tier, so "H2" -> 2.
SILICON_RUNGS = ("H0", "H1", "H2", "H3", "H4", "H5")
_SILICON_INDEX = {label: index for index, label in enumerate(SILICON_RUNGS)}


@dataclass(frozen=True, slots=True)
class CompletenessTiers:
    """A model's readiness on both catalogue-to-silicon axes.

    Parameters
    ----------
    science:
        Science-axis tier in ``0``-``5`` (S0-S5).
    silicon:
        Silicon-axis tier in ``0``-``5`` (H0-H5), or ``None`` when the model has
        no committed silicon evidence yet.
    """

    science: int
    silicon: int | None

    @property
    def science_label(self) -> str:
        """The science tier as an ``S<n>`` label."""
        return f"S{self.science}"

    @property
    def silicon_label(self) -> str:
        """The silicon tier as an ``H<n>`` label, or ``"none"`` when unattempted."""
        return "none" if self.silicon is None else f"H{self.silicon}"


def science_tier(descriptor: ModelDescriptor) -> int:
    """Return the science-axis tier (0-5) the descriptor reaches.

    S0-S3 are the curation kernel from
    :func:`~sc_neurocore.neurons.model_descriptor.descriptor_completeness_tier`.
    A descriptor only climbs above S3 when its evidence facets carry the proof:

    * **S4** — faithful dynamics: declared dynamics plus a confirmed three-way
      agreement with the publication (``validation.dynamics_faithful``).
    * **S5** — class-validated: a non-trivial metric and committed evidence
      (``validation.is_class_validated``).

    Parameters
    ----------
    descriptor:
        The model descriptor to score.

    Returns
    -------
    int
        The science tier in ``0``-``5``.
    """
    base = descriptor_completeness_tier(descriptor)
    if base < 3:
        return base
    if not (descriptor.dynamics and descriptor.validation.dynamics_faithful):
        return 3
    if not descriptor.validation.is_class_validated:
        return 4
    return 5


def silicon_tier(descriptor: ModelDescriptor) -> int | None:
    """Return the silicon-axis tier (0-5) the descriptor reaches, or ``None``.

    ``None`` means the model has no compile-clean RTL yet (no silicon evidence).
    Otherwise the ladder climbs one rung at a time, each rung credited only when
    both its boolean flag and its proof anchor are present:

    * **H0** — ``silicon.compiles`` (iverilog-valid RTL).
    * **H1** — Python<->Verilog validated (``cosim_validated`` + ``cosim_evidence``).
    * **H2** — synthesisable (``synthesised`` + ``synth_report``).
    * **H3** — timing-closed and resource-characterised (``timing_closed`` +
      ``timing_report`` + ``clock_mhz``).
    * **H4** — formally equivalent (``formally_equivalent`` + ``equivalence_proof``).
    * **H5** — tool-level signed PPA (``ppa_signed`` + ``ppa_report``).

    Parameters
    ----------
    descriptor:
        The model descriptor to score.

    Returns
    -------
    int | None
        The silicon tier in ``0``-``5``, or ``None`` when no RTL compiles.
    """
    silicon = descriptor.silicon
    if not silicon.compiles:
        return None
    if not (silicon.cosim_validated and silicon.cosim_evidence):
        return 0
    if not (silicon.synthesised and silicon.synth_report):
        return 1
    if not (silicon.timing_closed and silicon.timing_report and silicon.clock_mhz is not None):
        return 2
    if not (silicon.formally_equivalent and silicon.equivalence_proof):
        return 3
    if not (silicon.ppa_signed and silicon.ppa_report):
        return 4
    return 5


def completeness_tiers(descriptor: ModelDescriptor) -> CompletenessTiers:
    """Return both axis tiers for a descriptor.

    Parameters
    ----------
    descriptor:
        The model descriptor to score.

    Returns
    -------
    CompletenessTiers
        The science (0-5) and silicon (0-5 or ``None``) tiers together.
    """
    return CompletenessTiers(
        science=science_tier(descriptor),
        silicon=silicon_tier(descriptor),
    )


def is_perfect(descriptor: ModelDescriptor) -> bool:
    """Return whether a model is *perfect* by the master-plan acceptance contract.

    A model is perfect when it reaches S5 on the science axis and meets the
    terminal silicon tier its deployability class declares
    (``silicon.target_tier``). A model whose terminal tier is undeclared cannot
    be certified perfect: without the deployability class the required silicon
    tier is unknown, so this returns ``False`` rather than guessing.

    Parameters
    ----------
    descriptor:
        The model descriptor to judge.

    Returns
    -------
    bool
        ``True`` only when S5 is reached and the declared terminal H-tier is met.
    """
    if science_tier(descriptor) != 5:
        return False
    target = descriptor.silicon.target_tier
    if target not in _SILICON_INDEX:
        return False
    reached = silicon_tier(descriptor)
    return reached is not None and reached >= _SILICON_INDEX[target]


__all__ = [
    "SILICON_RUNGS",
    "CompletenessTiers",
    "completeness_tiers",
    "is_perfect",
    "science_tier",
    "silicon_tier",
]

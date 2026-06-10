# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — On-chip learning export

"""On-chip learning configuration utilities for neuromorphic targets.

Generates parameters and calibration files for platforms with in-situ
plasticity (BrainChip Akida, BrainScaleS, SpiNNaker).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OnChipLearningParams:
    """Parameters for on-chip STDP / reward-modulated plasticity.

    Attributes
    ----------
    learning_rule : str
        ``"stdp"``, ``"rstdp"`` (reward-modulated), or ``"triplet"``.
    tau_plus_ms : float
        Pre→post time constant (ms).
    tau_minus_ms : float
        Post→pre time constant (ms).
    a_plus : float
        Potentiation amplitude.
    a_minus : float
        Depression amplitude.
    w_max : float
        Maximum synaptic weight.
    w_min : float
        Minimum synaptic weight.
    reward_tau_ms : float
        Reward signal time constant (ms), for RSTDP.
    target_platform : str
        Target neuromorphic platform.
    """

    learning_rule: str
    tau_plus_ms: float
    tau_minus_ms: float
    a_plus: float
    a_minus: float
    w_max: float
    w_min: float
    reward_tau_ms: float
    target_platform: str


def generate_learning_params(
    *,
    learning_rule: str = "stdp",
    tau_plus_ms: float = 20.0,
    tau_minus_ms: float = 20.0,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    w_max: float = 1.0,
    w_min: float = 0.0,
    reward_tau_ms: float = 200.0,
    target: str = "akida2",
) -> OnChipLearningParams:
    """Generate on-chip learning parameters for neuromorphic targets.

    Creates calibration parameters for platforms with in-situ
    plasticity (BrainChip Akida 2, BrainScaleS-2, SpiNNaker 2).

    Parameters
    ----------
    learning_rule : str
        ``"stdp"`` (spike-timing), ``"rstdp"`` (reward-modulated),
        or ``"triplet"`` (triplet-based STDP).
    tau_plus_ms : float
        LTP time constant.
    tau_minus_ms : float
        LTD time constant.
    a_plus : float
        Potentiation amplitude.
    a_minus : float
        Depression amplitude.
    w_max : float
        Weight ceiling.
    w_min : float
        Weight floor.
    reward_tau_ms : float
        Reward eligibility trace time constant.
    target : str
        Target platform name.

    Returns
    -------
    OnChipLearningParams
        Complete learning parameter set.
    """
    return OnChipLearningParams(
        learning_rule=learning_rule,
        tau_plus_ms=tau_plus_ms,
        tau_minus_ms=tau_minus_ms,
        a_plus=a_plus,
        a_minus=a_minus,
        w_max=w_max,
        w_min=w_min,
        reward_tau_ms=reward_tau_ms,
        target_platform=target,
    )


def export_learning_config(
    params: OnChipLearningParams,
    *,
    output_format: str = "json",
) -> str:
    """Export on-chip learning parameters as a configuration file.

    Parameters
    ----------
    params : OnChipLearningParams
        Learning parameters from ``generate_learning_params()``.
    output_format : str
        ``"json"`` or ``"yaml"``.

    Returns
    -------
    str
        Configuration file content.
    """
    import json

    data = {
        "learning_rule": params.learning_rule,
        "time_constants": {
            "tau_plus_ms": params.tau_plus_ms,
            "tau_minus_ms": params.tau_minus_ms,
            "reward_tau_ms": params.reward_tau_ms,
        },
        "amplitudes": {
            "a_plus": params.a_plus,
            "a_minus": params.a_minus,
        },
        "weight_bounds": {
            "w_max": params.w_max,
            "w_min": params.w_min,
        },
        "target_platform": params.target_platform,
    }

    if output_format == "json":
        return json.dumps(data, indent=2)
    if output_format == "yaml":
        lines = ["# SC-NeuroCore On-Chip Learning Configuration"]
        lines.append(f"learning_rule: {params.learning_rule}")
        lines.append("time_constants:")
        lines.append(f"  tau_plus_ms: {params.tau_plus_ms}")
        lines.append(f"  tau_minus_ms: {params.tau_minus_ms}")
        lines.append(f"  reward_tau_ms: {params.reward_tau_ms}")
        lines.append("amplitudes:")
        lines.append(f"  a_plus: {params.a_plus}")
        lines.append(f"  a_minus: {params.a_minus}")
        lines.append("weight_bounds:")
        lines.append(f"  w_max: {params.w_max}")
        lines.append(f"  w_min: {params.w_min}")
        lines.append(f"target_platform: {params.target_platform}")
        return "\n".join(lines)
    raise ValueError(f"Unsupported learning config format: {output_format!r}")

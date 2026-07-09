#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UltraScale+ dense folding planner

"""Dense-layer folding planner for resource-bounded UltraScale+ targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DenseFoldPlan:
    n_inputs: int
    n_outputs: int
    mac_count: int
    dsp_budget: int
    output_parallelism: int
    input_parallelism: int
    dsp_per_cycle: int
    input_fold_factor: int
    output_fold_factor: int
    compute_cycles: int
    fold_required: bool
    fits_dsp_budget: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def _ceil_div(value: int, divisor: int) -> int:
    return 0 if value == 0 else 1 + (value - 1) // divisor


def plan_dense_fold(n_inputs: int, n_outputs: int, dsp_budget: int) -> DenseFoldPlan:
    """Plan a deterministic row-group fold for a dense matrix multiply.

    The planner preserves complete input rows when the row width fits the DSP
    budget. For the 64x32 ZU3EG case this maps five output rows per cycle,
    using 320 DSPs out of the 360-DSP budget and completing in seven cycles.
    """

    if n_inputs < 0 or n_outputs < 0 or dsp_budget < 0:
        raise ValueError("n_inputs, n_outputs, and dsp_budget must be non-negative")
    mac_count = n_inputs * n_outputs
    if n_inputs == 0 or n_outputs == 0 or dsp_budget == 0:
        return DenseFoldPlan(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            mac_count=mac_count,
            dsp_budget=dsp_budget,
            output_parallelism=0,
            input_parallelism=0,
            dsp_per_cycle=0,
            input_fold_factor=0,
            output_fold_factor=0,
            compute_cycles=0,
            fold_required=mac_count > dsp_budget,
            fits_dsp_budget=False,
        )

    if dsp_budget >= n_inputs:
        output_parallelism = min(n_outputs, max(1, dsp_budget // n_inputs))
    else:
        output_parallelism = 1
    input_parallelism = min(n_inputs, max(1, dsp_budget // output_parallelism))
    dsp_per_cycle = output_parallelism * input_parallelism
    input_fold_factor = _ceil_div(n_inputs, input_parallelism)
    output_fold_factor = _ceil_div(n_outputs, output_parallelism)
    compute_cycles = input_fold_factor * output_fold_factor
    return DenseFoldPlan(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        mac_count=mac_count,
        dsp_budget=dsp_budget,
        output_parallelism=output_parallelism,
        input_parallelism=input_parallelism,
        dsp_per_cycle=dsp_per_cycle,
        input_fold_factor=input_fold_factor,
        output_fold_factor=output_fold_factor,
        compute_cycles=compute_cycles,
        fold_required=mac_count > dsp_budget,
        fits_dsp_budget=dsp_per_cycle <= dsp_budget,
    )

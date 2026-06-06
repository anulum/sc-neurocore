#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - DCLS Q8.8 cosimulation reference

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

FRACTION = 8
Q88_ONE = 1 << FRACTION
I16_MAX_Q16_16 = 32767 << FRACTION
I16_MIN_Q16_16 = -32768 << FRACTION
I32_MAX = 2_147_483_647
I32_MIN = -2_147_483_648


def tent_gate_q88(tap_index: int, centre_q88: int, sigma_q88: int) -> int:
    if sigma_q88 <= 0:
        raise ValueError(f"DCLS tent sigma must be positive, got {sigma_q88}")
    delay_q88 = tap_index << FRACTION
    distance_q88 = abs(delay_q88 - centre_q88)
    if distance_q88 >= sigma_q88:
        return 0
    return min(Q88_ONE, max(0, ((sigma_q88 - distance_q88) << FRACTION) // sigma_q88))


def dcls_q88_reference(
    spikes: list[int], weights_q88: list[int], centre_q88: int, sigma_q88: int
) -> dict[str, int | bool]:
    if not spikes:
        raise ValueError("DCLS forward pass requires at least one tap")
    if len(spikes) != len(weights_q88):
        raise ValueError(
            f"DCLS spike/weight length mismatch: spikes={len(spikes)}, weights={len(weights_q88)}"
        )
    accumulator = 0
    active_tap_count = 0
    max_gate_q88 = 0
    for tap_index, (spike, weight_q88) in enumerate(zip(spikes, weights_q88, strict=True)):
        if spike == 0:
            continue
        active_tap_count += 1
        gate_q88 = tent_gate_q88(tap_index, centre_q88, sigma_q88)
        max_gate_q88 = max(max_gate_q88, gate_q88)
        accumulator += int(weight_q88) * gate_q88

    accumulator_q16_16 = min(I32_MAX, max(I32_MIN, accumulator))
    overflow = accumulator_q16_16 != accumulator
    if accumulator > I16_MAX_Q16_16:
        output_q88 = 32767
        overflow = True
    elif accumulator < I16_MIN_Q16_16:
        output_q88 = -32768
        overflow = True
    else:
        output_q88 = accumulator >> FRACTION
    return {
        "output_q88": int(output_q88),
        "accumulator_q16_16": int(accumulator_q16_16),
        "overflow": bool(overflow),
        "active_tap_count": active_tap_count,
        "max_gate_q88": int(max_gate_q88),
    }


def dcls_q88_pytorch_reference(
    spikes: list[int], weights_q88: list[int], centre_q88: int, sigma_q88: int
) -> dict[str, int | bool]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the DCLS PyTorch reference") from exc

    if sigma_q88 <= 0:
        raise ValueError(f"DCLS tent sigma must be positive, got {sigma_q88}")
    spike_tensor = torch.tensor(spikes, dtype=torch.int64)
    weight_tensor = torch.tensor(weights_q88, dtype=torch.int64)
    tap_indices = torch.arange(len(spikes), dtype=torch.int64)
    delay_q88 = tap_indices << FRACTION
    distance_q88 = torch.abs(delay_q88 - int(centre_q88))
    numerator = torch.clamp(int(sigma_q88) - distance_q88, min=0)
    gate_q88 = torch.div(numerator << FRACTION, int(sigma_q88), rounding_mode="trunc")
    gate_q88 = torch.clamp(gate_q88, min=0, max=Q88_ONE)
    accumulator = int(torch.sum(spike_tensor * weight_tensor * gate_q88).item())
    accumulator_q16_16 = min(I32_MAX, max(I32_MIN, accumulator))
    overflow = accumulator_q16_16 != accumulator
    if accumulator > I16_MAX_Q16_16:
        output_q88 = 32767
        overflow = True
    elif accumulator < I16_MIN_Q16_16:
        output_q88 = -32768
        overflow = True
    else:
        output_q88 = accumulator >> FRACTION
    active = int(torch.count_nonzero(spike_tensor).item())
    max_gate = int(torch.max(gate_q88 * (spike_tensor != 0)).item()) if spikes else 0
    return {
        "output_q88": int(output_q88),
        "accumulator_q16_16": int(accumulator_q16_16),
        "overflow": bool(overflow),
        "active_tap_count": active,
        "max_gate_q88": max_gate,
    }


def deterministic_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "hand_computed_three_tap",
            "spikes": [1, 1, 1],
            "weights_q88": [256, 128, -64],
            "centre_q88": 256,
            "sigma_q88": 512,
        },
        {
            "name": "silent_taps",
            "spikes": [0, 1, 0, 1],
            "weights_q88": [512, -256, 64, 128],
            "centre_q88": 512,
            "sigma_q88": 768,
        },
        {
            "name": "outside_tent_zeroes_far_taps",
            "spikes": [1, 1, 1, 1, 1],
            "weights_q88": [16, 32, 64, 128, 256],
            "centre_q88": 0,
            "sigma_q88": 256,
        },
        {
            "name": "negative_contribution",
            "spikes": [1, 1, 1, 1],
            "weights_q88": [-512, -128, 256, 64],
            "centre_q88": 256,
            "sigma_q88": 1024,
        },
        {
            "name": "output_saturation",
            "spikes": [1] * 128,
            "weights_q88": [32767] * 128,
            "centre_q88": 0,
            "sigma_q88": 32767,
        },
    ]


def run_deterministic_suite(require_torch: bool = True) -> dict[str, Any]:
    cases = deterministic_cases()
    torch_available = True
    comparisons: list[dict[str, Any]] = []
    max_abs_diff = 0
    for case in cases:
        python_result = dcls_q88_reference(
            case["spikes"], case["weights_q88"], case["centre_q88"], case["sigma_q88"]
        )
        torch_result: dict[str, int | bool] | None = None
        try:
            torch_result = dcls_q88_pytorch_reference(
                case["spikes"], case["weights_q88"], case["centre_q88"], case["sigma_q88"]
            )
        except RuntimeError:
            torch_available = False
            if require_torch:
                raise
        if torch_result is not None:
            diff = abs(
                int(python_result["accumulator_q16_16"]) - int(torch_result["accumulator_q16_16"])
            )
            max_abs_diff = max(max_abs_diff, diff)
            passed = python_result == torch_result
        else:
            diff = 0
            passed = True
        comparisons.append(
            {
                "name": case["name"],
                "python": python_result,
                "pytorch": torch_result,
                "abs_accumulator_diff": diff,
                "passed": passed,
            }
        )
    cases_passed = sum(1 for item in comparisons if item["passed"])
    return {
        "case_count": len(cases),
        "cases_passed": cases_passed,
        "max_abs_accumulator_diff": max_abs_diff,
        "pytorch_available": torch_available,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DCLS Q8.8 Python and PyTorch references.")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-missing-torch", action="store_true")
    args = parser.parse_args()
    report = run_deterministic_suite(require_torch=not args.allow_missing_torch)
    if report["cases_passed"] != report["case_count"]:
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

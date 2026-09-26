# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Comparable operator-supplied measurement receipts

"""Compare supplied measurements only under one explicit acquisition contract.

This validates document custody and comparability, not an instrument's physical
accuracy. No energy or latency is estimated from simulation work counts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from sc_neurocore.fitting.cohort import cohort_sha256

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_FIELDS = frozenset(
    {
        "target",
        "device_revision",
        "harness_sha256",
        "workload_sha256",
        "warmup",
        "transport",
        "repeats",
        "aggregation",
        "resource_unit",
        "instrument",
        "calibration_sha256",
    }
)


def measured_pareto(
    result: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return nondominated holdout-error/latency/resource/energy rows or refuse.

    Parameters
    ----------
    result:
        Complete cohort result whose scientific digest and trials are verified.
    receipts:
        Operator-supplied physical measurement documents, each bound to one
        trial and the cohort. All acquisition contracts must match exactly.

    Returns
    -------
    dict
        Custody-labelled comparison; all four axes are minimised. Missing or
        incomparable evidence yields no frontier and an explicit reason.
    """
    base = {
        "schema_version": "sc-neurocore.measured-pareto.v1",
        "custody": "operator-supplied receipts; instrument accuracy is not independently verified",
    }
    try:
        scientific = {key: value for key, value in result.items() if key != "result_sha256"}
        if (
            result.get("schema_version") != "sc-neurocore.cohort-result.v1"
            or cohort_sha256(scientific) != result["result_sha256"]
        ):
            raise ValueError("cohort result digest does not match")
        if len(receipts) < 2:
            raise ValueError("at least two comparable measured trials are required")
        trials = {trial["trial_sha256"]: trial for trial in result["trials"]}
        contracts: set[str] = set()
        metrics: set[str] = set()
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for receipt in receipts:
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if (
                receipt.get("schema_version") != "sc-neurocore.measurement.v1"
                or receipt.get("source_kind") != "physical"
                or cohort_sha256(unsigned) != receipt.get("receipt_sha256")
            ):
                raise ValueError("a measurement needs a digest-bound physical receipt")
            if receipt["cohort_sha256"] != result["provenance"]["cohort_sha256"]:
                raise ValueError("measurement belongs to a different cohort")
            trial_id = str(receipt["trial_sha256"])
            if trial_id in seen:
                raise ValueError("duplicate trial measurement")
            seen.add(trial_id)
            trial = trials[trial_id]
            if trial["status"] != "completed":
                raise ValueError("failed trials cannot enter a measurement frontier")
            contract = receipt["contract"]
            if set(contract) != _CONTRACT_FIELDS or any(
                not isinstance(contract[field], str) or not contract[field].strip()
                for field in _CONTRACT_FIELDS - {"repeats"}
            ):
                raise ValueError("measurement acquisition contract is incomplete")
            if (
                not isinstance(contract["repeats"], int)
                or isinstance(contract["repeats"], bool)
                or contract["repeats"] < 1
            ):
                raise ValueError("measurement repeats must be a positive integer")
            for field in ("harness_sha256", "workload_sha256", "calibration_sha256"):
                if not _HEX.fullmatch(contract[field]):
                    raise ValueError("measurement provenance needs SHA-256 digests")
            if contract["workload_sha256"] != result["provenance"]["cohort_sha256"]:
                raise ValueError("measurement workload must bind the complete shared-sample cohort")
            contracts.add(cohort_sha256(contract))
            metrics.add(cohort_sha256(trial["metric"]))
            held = [sample["value"] for sample in trial["samples"] if sample["split"] == "holdout"]
            axes = [
                sum(held) / len(held),
                receipt["latency_ms"],
                receipt["resources"],
                receipt["energy_j"],
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in axes
            ):
                raise ValueError("measured axes must be finite nonnegative values")
            rows.append(
                {
                    "trial_sha256": trial_id,
                    "model": trial["model"],
                    "holdout_error": axes[0],
                    "latency_ms": axes[1],
                    "resources": axes[2],
                    "energy_j": axes[3],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "axes": axes,
                }
            )
        if len(contracts) != 1 or len(metrics) != 1:
            raise ValueError("acquisition contracts and scientific metric units must match")
        for row in rows:
            row["nondominated"] = not any(
                all(a <= b for a, b in zip(other["axes"], row["axes"], strict=True))
                and any(a < b for a, b in zip(other["axes"], row["axes"], strict=True))
                for other in rows
            )
        for row in rows:
            row.pop("axes")
        return {
            **base,
            "comparable": True,
            "contract": dict(receipts[0]["contract"]),
            "metric": dict(trials[str(receipts[0]["trial_sha256"])]["metric"]),
            "rows": rows,
            "reason": None,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {**base, "comparable": False, "rows": [], "reason": str(exc)}

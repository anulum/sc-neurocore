# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Generate the source-bound SC Compte behavior ensemble receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from sc_neurocore.network import (
    SC_COMPTE_WM_BEHAVIOR_BACKENDS,
    SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS,
    SCCompteWMBehaviorAcceptance,
    SCCompteWMBehaviorProtocol,
    SCCompteWMBehaviorTrial,
    run_sc_compte_wm_behavior_trial,
    summarize_sc_compte_wm_behavior_ensemble,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_behavior_ensemble.json"
SOURCE_PATHS = (
    "benchmarks/bench_sc_compte_wm_behavior_ensemble.py",
    "src/sc_neurocore/network/sc_compte_wm.py",
    "src/sc_neurocore/network/sc_compte_wm_drive.py",
    "src/sc_neurocore/network/sc_compte_wm_network.py",
    "src/sc_neurocore/network/sc_compte_wm_backends.py",
    "src/sc_neurocore/network/sc_compte_wm_behavior.py",
    "engine/src/sc_compte_wm_network.rs",
    "engine/examples/sc_compte_wm_network_run.rs",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/SCCompteWMNetwork.jl",
    "src/sc_neurocore/accel/julia/sc_compte_wm_network/run_sc_compte_wm_network.jl",
    "src/sc_neurocore/accel/go/sc_compte_wm_network/network.go",
    "src/sc_neurocore/accel/go/cmd/run_sc_compte_wm_network/main.go",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/sc_compte_wm_network.mojo",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/libsc_compte_wm_network.so",
    "src/sc_neurocore/accel/mojo/sc_compte_wm_network/__init__.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trial_payload(trial: SCCompteWMBehaviorTrial) -> dict[str, Any]:
    payload = asdict(trial)
    payload["checks"] = {name: passed for name, passed in trial.checks}
    return payload


def build_payload(timeout_s: float) -> dict[str, Any]:
    """Run three Rust seeds plus one anchor on every other explicit backend."""
    protocol = SCCompteWMBehaviorProtocol()
    acceptance = SCCompteWMBehaviorAcceptance()
    trials = [
        run_sc_compte_wm_behavior_trial(
            backend="rust",
            seed=seed,
            protocol=protocol,
            acceptance=acceptance,
            timeout_s=timeout_s,
        )
        for seed in SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS
    ]
    trials.extend(
        run_sc_compte_wm_behavior_trial(
            backend=backend,
            seed=42,
            protocol=protocol,
            acceptance=acceptance,
            timeout_s=timeout_s,
        )
        for backend in SC_COMPTE_WM_BEHAVIOR_BACKENDS
        if backend != "rust"
    )
    ensemble = summarize_sc_compte_wm_behavior_ensemble(tuple(trials), acceptance=acceptance)
    return {
        "schema_version": "sc-neurocore.sc-compte-wm-behavior-ensemble.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "SC-COMPTE-WM-NETWORK",
        "evidence_class": "deterministic_simulator_behavior_ensemble",
        "source_reproduction_claimed": False,
        "hardware_measurement_claimed": False,
        "production_speed_claimed": False,
        "persistent_bump_claimed": ensemble.passed,
        "distractor_resistance_claimed": ensemble.passed,
        "response_reset_claimed": ensemble.passed,
        "protocol": asdict(protocol),
        "acceptance": asdict(acceptance),
        "reference": {
            "backend": ensemble.reference_backend,
            "seeds": ensemble.reference_seeds,
        },
        "source_sha256": {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS},
        "trials": [_trial_payload(trial) for trial in trials],
        "ensemble": asdict(ensemble),
        "passed": ensemble.passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.timeout_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

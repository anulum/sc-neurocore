# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Butera Model 1 benchmark evidence gate

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "benchmarks/results/bench_butera_respiratory.json"
_SOURCES = {
    "benchmarks/bench_model_butera_respiratory.py": _ROOT
    / "benchmarks/bench_model_butera_respiratory.py",
    "engine/Cargo.toml": _ROOT / "engine/Cargo.toml",
    "engine/examples/bench_butera_respiratory_rk4.rs": _ROOT
    / "engine/examples/bench_butera_respiratory_rk4.rs",
    "engine/src/neurons/simple_spiking.rs": _ROOT / "engine/src/neurons/simple_spiking.rs",
    "engine/src/neurons/simple_spiking/reexports.rs": _ROOT
    / "engine/src/neurons/simple_spiking/reexports.rs",
    "engine/src/neurons/simple_spiking/butera_respiratory.rs": _ROOT
    / "engine/src/neurons/simple_spiking/butera_respiratory.rs",
    "src/sc_neurocore/neurons/models/butera_respiratory.py": _ROOT
    / "src/sc_neurocore/neurons/models/butera_respiratory.py",
    "src/sc_neurocore/accel/go/services/butera_respiratory.go": _ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory.go",
    "src/sc_neurocore/accel/go/services/butera_respiratory_test.go": _ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory_test.go",
    "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl": _ROOT
    / "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl",
    "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo": _ROOT
    / "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo",
    "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs": _ROOT
    / "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs",
}


def test_result_is_bound_to_all_runtime_and_engine_sources() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    expected = {
        source: hashlib.sha256(path.read_bytes()).hexdigest() for source, path in _SOURCES.items()
    }
    assert payload["source_hashes"] == expected


def test_result_records_five_runtime_event_parity_without_speed_claim() -> None:
    payload = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert (payload["steps"], payload["current"]) == (200_000, 50.0)
    assert set(payload["backend_summary"]) == {"python", "rust", "go", "julia", "mojo"}
    assert len({entry["spikes"] for entry in payload["backend_summary"].values()}) == 1
    assert payload["hardware_measurement_claimed"] is False
    assert payload["production_speed_claim"] is False

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust network and safety source contracts

"""Compile the dependency-free Rust safety oracle and enforce native docs."""

from __future__ import annotations

from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]
SAFETY = REPOSITORY / "src/sc_neurocore/accel/rust/safety/sc_compte_wm_network.rs"
NATIVE = REPOSITORY / "engine/src/sc_compte_wm_network.rs"


def test_dependency_free_safety_oracle_executes(tmp_path: Path) -> None:
    binary = tmp_path / "sc-compte-wm-network-safety"
    compiled = subprocess.run(
        ["rustc", "--edition", "2021", "--test", str(SAFETY), "-o", str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=60
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "3 passed" in executed.stdout


def test_public_native_network_surface_has_rustdoc() -> None:
    source = NATIVE.read_text(encoding="utf-8")
    assert "//! Full 2,560-cell Rust runtime" in source
    for symbol in (
        "pub struct SCCompteWMNetworkSpec",
        "pub struct SCCompteWMNetworkState",
        "pub struct SCCompteWMStepReceipt",
        "pub struct SCCompteWMNetwork",
        "pub fn counter_poisson_counts",
        "pub fn step_with_events",
    ):
        offset = source.index(symbol)
        assert "///" in source[max(0, offset - 500) : offset]

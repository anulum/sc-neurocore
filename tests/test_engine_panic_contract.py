# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine panic and overflow contract tests

"""Hold the engine's release contract: panics unwind, documented panics are refused first, overflow wraps.

The release profile set ``panic = "abort"``, so a panic in the engine ended the Python process. It
now unwinds. The two documented panics Python could reach are refused in their bindings before the
Rust code runs: a hybrid linear-attention neuron of no key-value slots, whose step divides by zero,
and an allocation failure while copying a differentiable dense layer's weights. Each refusal runs
in a child interpreter, so a regression shows as a dead child rather than a dead test run.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _child(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )


def test_the_release_profile_unwinds_and_leaves_overflow_unchecked() -> None:
    """The wheels' profile unwinds on a panic and wraps on overflow, as the engine page states."""
    release = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["profile"]["release"]
    assert release["panic"] == "unwind"
    assert "overflow-checks" not in release
    page = (ROOT / "docs/api/rust-engine.md").read_text(encoding="utf-8")
    assert "## Panics and integer overflow in release builds" in page
    assert 'sets `panic = "unwind"` and no `overflow-checks`' in page
    assert "**Integer overflow wraps.**" in page


def test_a_hybrid_attention_neuron_of_no_slots_is_refused_in_a_live_interpreter() -> None:
    """``dim=0`` raises ``ValueError`` before the step could divide by zero; ``dim=1`` steps."""
    result = _child(
        "from sc_neurocore_engine.sc_neurocore_engine import RustHybridLinearAttentionNeuron as N\n"
        "try:\n"
        "    N(0)\n"
        "except ValueError as error:\n"
        "    print('refused:', error)\n"
        "else:\n"
        "    raise SystemExit('dim=0 was accepted')\n"
        "print('stepped:', N(1).step_qkv(0.5, 0.25, 1.0))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "refused: dim must be at least 1" in result.stdout
    assert "stepped:" in result.stdout


def test_a_differentiable_dense_layer_copies_its_weights_through_the_fallible_path() -> None:
    """``get_weights`` keeps its result, one row per neuron, now through the copy that cannot panic."""
    result = _child(
        "from sc_neurocore_engine.sc_neurocore_engine import DifferentiableDenseLayer as L\n"
        "weights = L(3, 2).get_weights()\n"
        "print(len(weights), sorted({len(row) for row in weights}))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[0] == "2 [3]"
    binding = (ROOT / "engine/src/bindings/differentiable_dense.rs").read_text(encoding="utf-8")
    assert ".try_get_weights()" in binding and ".get_weights()\n" not in binding

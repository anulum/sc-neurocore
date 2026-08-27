# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel benchmark dynamics-parity gate

"""Bind every published Brunel-vs-Brian2 speed claim to dynamics parity.

The committed scaling artefact carries three simulator lanes. Only a lane
whose firing rate tracks the Brian2 reference at matched network sizes may be
presented as a like-for-like comparison; the fixed-point Rust lane is an
unconnected LIF layer whose rate does not scale with the network, so it is
pinned here as a non-parity lane and the public pages must say so wherever
they cite its wall-clock ratios.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ARTEFACT = _ROOT / "benchmarks/results/rust_scaling_benchmark.json"

# Simulator lanes whose dynamics are validated against Brian2 at matched sizes.
_PARITY_SIMULATORS = frozenset({"sc_numpy_dense"})
# Lanes recorded in the artefact that are NOT a dynamics-parity comparison.
# Moving a lane out of this set requires a new artefact whose rows actually
# track the Brian2 rates — never a wording change alone.
_NON_PARITY_SIMULATORS = frozenset({"sc_rust_engine"})

_RATE_TOLERANCE = 0.10
_PUBLIC_CLAIM_FILES = (
    "README.md",
    "docs/COMPETITIVE_LANDSCAPE.md",
    "docs/guides/faq.md",
)


def _rows() -> dict[tuple[str, int], dict[str, float]]:
    payload = json.loads(_ARTEFACT.read_text(encoding="utf-8"))
    return {(row["simulator"], row["n_neurons"]): row for row in payload["data"]}


def test_artefact_classifies_every_simulator_lane() -> None:
    """Every recorded lane is explicitly parity, non-parity, or the reference."""

    simulators = {simulator for simulator, _ in _rows()}
    assert simulators == {"brian2"} | _PARITY_SIMULATORS | _NON_PARITY_SIMULATORS


def test_parity_lane_tracks_brian2_rates_at_matched_sizes() -> None:
    """The parity lane must reproduce the Brian2 rate envelope within 10%."""

    rows = _rows()
    for simulator in sorted(_PARITY_SIMULATORS):
        matched = 0
        for (candidate, n_neurons), row in rows.items():
            if candidate != simulator or ("brian2", n_neurons) not in rows:
                continue
            reference = rows[("brian2", n_neurons)]
            ratio = row["rate_mean_hz"] / reference["rate_mean_hz"]
            assert abs(ratio - 1.0) <= _RATE_TOLERANCE, (
                f"{simulator} at N={n_neurons} fires at {ratio:.3f}x the Brian2 rate"
            )
            matched += 1
        assert matched >= 3, f"{simulator} needs at least three matched sizes"


def test_non_parity_lane_remains_documented_as_non_parity() -> None:
    """The unconnected fixed-point lane must not silently become a parity claim."""

    rows = _rows()
    for simulator in sorted(_NON_PARITY_SIMULATORS):
        for (candidate, n_neurons), row in rows.items():
            if candidate != simulator or ("brian2", n_neurons) not in rows:
                continue
            reference = rows[("brian2", n_neurons)]
            ratio = row["rate_mean_hz"] / reference["rate_mean_hz"]
            assert abs(ratio - 1.0) > _RATE_TOLERANCE, (
                f"{simulator} at N={n_neurons} now tracks Brian2 ({ratio:.3f}x): "
                "re-classify the lane instead of leaving it in the non-parity set"
            )


def test_public_pages_bind_the_speed_claim_to_the_parity_boundary() -> None:
    """Every public 39-202x citation must carry the non-parity disclaimer."""

    for relative in _PUBLIC_CLAIM_FILES:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert "39-202x" in text, f"{relative} no longer cites the artefact ratios"
        assert "not a dynamics-parity result" in text, (
            f"{relative} cites 39-202x without the non-parity disclaimer"
        )

    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "39-202x vs Brian2" not in readme, (
        "README presents the non-parity wall-clock ratio as a bare Brian2 comparison"
    )
    assert "2.4x" in readme, "README omits the parity-validated speed numbers"

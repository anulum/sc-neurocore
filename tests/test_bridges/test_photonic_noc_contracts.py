# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Edge contracts for the photonic Network-on-Chip bridge."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

import sc_neurocore.bridges.photonic_noc as photonic_noc
from tests.module_reload import preserve_module_identity


def test_optional_gdstk_import_branch_records_available_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload the bridge with a present GDS module and restore import state."""
    fake_gdstk = ModuleType("gdstk")
    monkeypatch.setitem(sys.modules, "gdstk", fake_gdstk)

    # Restore the module's original class identities on exit; a bare reload-to-restore
    # would leave fresh classes that sibling modules' by-value imports fail isinstance on.
    with preserve_module_identity(photonic_noc):
        importlib.reload(photonic_noc)

        assert photonic_noc._HAS_GDSTK is True
        assert vars(photonic_noc)["gdstk"] is fake_gdstk


def test_power_budget_counts_failed_detector_paths() -> None:
    """Fail closed when accumulated optical loss drops below sensitivity."""
    design = photonic_noc.PhotonicCircuitDesign(
        name="lossy",
        waveguides=[
            photonic_noc.WaveguideSegment(
                source=0,
                target=1,
                length_um=1_000.0,
                loss_db=35.0,
            )
        ],
        mzi_gates=[
            photonic_noc.MZIGate(
                gate_id="mzi_loss",
                input_ports=[0],
                output_port=1,
                insertion_loss_db=2.0,
            )
        ],
        wdm_channels=[],
        n_nodes=2,
    )

    result = photonic_noc.PowerBudgetAnalyzer().analyze(design)

    assert result["n_failed"] == 1
    assert result["paths"][0]["passed"] is False
    assert result["paths"][0]["margin_db"] < 0.0


def test_visualization_reports_truncated_waveguide_inventory() -> None:
    """Keep long photonic inventories readable by reporting omitted paths."""
    design = photonic_noc.PhotonicCircuitDesign(
        name="long_waveguide_inventory",
        waveguides=[
            photonic_noc.WaveguideSegment(
                source=index,
                target=index + 1,
                length_um=100.0 + float(index),
                loss_db=0.1,
            )
            for index in range(11)
        ],
        mzi_gates=[],
        wdm_channels=[],
        n_nodes=12,
    )

    rendering = photonic_noc.visualize_photonic(design)

    assert "... and 1 more" in rendering

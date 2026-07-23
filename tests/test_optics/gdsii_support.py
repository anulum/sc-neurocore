# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gdsii.py

from __future__ import annotations

"""Tests for :meth:`CompilationResult.to_gdsii`.

The export turns a compiled MZI cascade into a real GDSII file via
``gdsfactory``. Tests cover the empty-layout guard, the layout dictionary
shape, the pitch/total-length arithmetic, and — when ``gdsfactory`` is
available — that the produced GDS file is non-empty and parses back.
"""
from collections.abc import Iterator
from pathlib import Path
import pytest
gf = pytest.importorskip(
    "gdsfactory",
    reason="gdsfactory is an optional dep (install via `pip install sc-neurocore[optics]`)",
)
from sc_neurocore.optics.photonic_emitter import CompilationResult  # noqa: E402
@pytest.fixture(autouse=True)
def _clear_gdsfactory_cache() -> Iterator[None]:
    """gdsfactory uses a process-wide KLayout layout registry — components
    with duplicate names across tests clash. Clear the cache before every
    test and reactivate the generic PDK so target names like
    ``silicon_photonics`` are free to reuse."""
    try:
        gf.clear_cache()
    except AttributeError:  # pragma: no cover - older/newer gdsfactory API split.
        # gdsfactory ≥ 9 uses kcl.clear()
        if hasattr(gf, "kcl"):
            gf.kcl.clear()
    # clear_cache drops the active PDK; reactivate the generic one so
    # gf.components.mzi() can resolve its default via get_active_pdk().
    try:
        gf.gpdk.PDK.activate()
    except AttributeError:  # pragma: no cover - compatibility path for older installs.
        try:
            from gdsfactory.generic_tech import get_generic_pdk

            get_generic_pdk().activate()
        except Exception:  # pragma: no cover - defensive global PDK reset.
            pass
    except Exception:  # pragma: no cover - defensive global PDK reset.
        pass
    yield
@pytest.fixture
def populated_result() -> CompilationResult:
    return CompilationResult(
        target="silicon_photonics",
        num_modulators=4,
        optical_power_mean_mw=1.25,
        phase_coverage_rad=3.14,
        netlist=(
            "# SC-NeuroCore photonic netlist\n"
            "module sc_photonic_top();\n"
            "  mzi m0(.phase(0.1));\n"
            "  mzi m1(.phase(0.6));\n"
            "  mzi m2(.phase(1.2));\n"
            "  mzi m3(.phase(2.0));\n"
            "endmodule\n"
        ),
    )

__all__ = ['Iterator', 'Path', 'pytest', 'gf', 'CompilationResult', '_clear_gdsfactory_cache', 'populated_result']

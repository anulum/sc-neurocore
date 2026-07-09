# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for GDSII export from photonic compilation result

"""Tests for :meth:`CompilationResult.to_gdsii`.

The export turns a compiled MZI cascade into a real GDSII file via
``gdsfactory``. Tests cover the empty-layout guard, the layout dictionary
shape, the pitch/total-length arithmetic, and — when ``gdsfactory`` is
available — that the produced GDS file is non-empty and parses back.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Empty-layout guard (must never silently write a layout with no cells).
# ---------------------------------------------------------------------------


class TestEmptyLayoutGuard:
    def test_zero_modulators_raises(self, tmp_path: Path) -> None:
        """Reject an empty physical layout instead of writing a silent shell."""
        empty = CompilationResult(
            target="x",
            num_modulators=0,
            optical_power_mean_mw=0.0,
            phase_coverage_rad=0.0,
            netlist="",
        )
        with pytest.raises(NotImplementedError, match="num_modulators > 0"):
            empty.to_gdsii(str(tmp_path / "empty.gds"))


# ---------------------------------------------------------------------------
# Layout arithmetic (pitch, total length, returned dict).
# ---------------------------------------------------------------------------


class TestLayoutArithmetic:
    def test_returns_full_layout_dict(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Return the emitted file path and geometric parameters for audit."""
        out_path = tmp_path / "cascade.gds"
        info = populated_result.to_gdsii(str(out_path), mzi_length_um=12.5, pitch_um=80.0)
        assert info["filename"] == str(out_path)
        assert info["n_modulators"] == populated_result.num_modulators
        assert info["mzi_length_um"] == pytest.approx(12.5)
        assert info["pitch_um"] == pytest.approx(80.0)
        assert info["target"] == populated_result.target
        # Four MZIs at 80 µm pitch ⇒ final origin at 4·80 = 320 µm.
        assert info["total_length_um"] == pytest.approx(4 * 80.0)

    def test_pitch_scales_total_length_linearly(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Scale reported cascade length linearly with MZI pitch."""
        info_a = populated_result.to_gdsii(str(tmp_path / "a.gds"), pitch_um=50.0)
        info_b = populated_result.to_gdsii(str(tmp_path / "b.gds"), pitch_um=200.0)
        assert info_b["total_length_um"] == pytest.approx(info_a["total_length_um"] * 4.0)


# ---------------------------------------------------------------------------
# Produced file is real GDSII.
# ---------------------------------------------------------------------------


class TestGDSFileRoundtrip:
    def test_file_created_and_nonempty(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Write a non-empty GDSII file for a populated cascade."""
        out_path = tmp_path / "roundtrip.gds"
        populated_result.to_gdsii(str(out_path))
        assert out_path.exists()
        size = out_path.stat().st_size
        assert size > 0

    def test_file_reads_back_via_gdsfactory(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Parse the exported GDSII file back through gdsfactory."""
        out_path = tmp_path / "parse.gds"
        populated_result.to_gdsii(str(out_path))
        # gdsfactory's low-level reader (via gdspy / klayout) must parse it.
        comp = gf.read.import_gds(str(out_path))
        assert comp is not None

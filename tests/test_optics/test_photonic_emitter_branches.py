# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Branch tests for photonic emitter helpers

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pytest

from sc_neurocore.optics.photonic_emitter import (
    BitstreamToOptical,
    CompilationResult,
    CrosstalkModel,
    FDTD2DSolver,
    MeepAdapter,
    OpticalModulation,
    PhotonicCompiler,
    PhotonicTarget,
    WaveguidePair,
)


def test_photonic_target_classmethods_cover_variants() -> None:
    assert PhotonicTarget.lightmatter().modulation is OpticalModulation.PHASE
    assert PhotonicTarget.silicon_photonics().modulator_type == "Microring"
    assert PhotonicTarget.two_d_waveguide().modulation is OpticalModulation.HYBRID


@pytest.mark.parametrize(
    ("target", "expected_phase", "expected_amp"),
    [
        (PhotonicTarget.lightmatter(), [0.0, math.pi], [1.0, 1.0]),
        (PhotonicTarget.silicon_photonics(), [0.0, 0.0], [1.0, 0.0]),
        (PhotonicTarget.two_d_waveguide(), [0.0, math.pi / 2], [1.0, 0.8]),
    ],
)
def test_bitstream_to_optical_variants(
    target: PhotonicTarget, expected_phase: list[float], expected_amp: list[float]
) -> None:
    converter = BitstreamToOptical(target)
    bits = np.array([1, 0], dtype=np.uint8)
    pulses = converter.convert(bits)
    assert [pulse.phase for pulse in pulses] == pytest.approx(expected_phase)
    assert [pulse.amplitude for pulse in pulses] == pytest.approx(expected_amp)
    np.testing.assert_allclose(converter.to_phase_array(bits), np.array(expected_phase))
    np.testing.assert_allclose(converter.to_amplitude_array(bits), np.array(expected_amp))


def test_optical_power_profile_applies_loss() -> None:
    target = PhotonicTarget.lightmatter()
    converter = BitstreamToOptical(target)
    profile = converter.optical_power_profile(np.array([1, 1], dtype=np.uint8), input_power_mw=2.0)
    expected = (10.0 ** (-target.insertion_loss_db / 10.0)) * 2.0
    np.testing.assert_allclose(profile, np.array([expected, expected]))


def test_photonic_compiler_empty_bitstream_rejected() -> None:
    compiler = PhotonicCompiler(PhotonicTarget.lightmatter())
    with pytest.raises(ValueError, match="cannot be empty"):
        compiler.compile_bitstream(np.array([], dtype=np.uint8))


def test_photonic_compiler_microring_netlist_and_fdtd() -> None:
    compiler = PhotonicCompiler(PhotonicTarget.silicon_photonics())
    result = compiler.compile_bitstream(
        np.array([1, 0, 1], dtype=np.uint8), run_fdtd=True, fdtd_steps=5
    )
    assert isinstance(result, CompilationResult)
    assert "MICRORING ring_0" in result.netlist
    assert result.num_modulators >= 1
    assert result.fdtd_energy >= 0.0


def test_meep_adapter_build_and_unavailable_run_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = PhotonicTarget.lightmatter()
    geometry = MeepAdapter.build_waveguide_geometry(target, waveguide_width_um=0.6, length_um=12.0)
    assert geometry["cell_size"][0] == 12.0
    assert geometry["sources"][0]["type"] == "ContinuousSource"
    monkeypatch.setitem(sys.modules, "meep", None)
    with pytest.raises(ImportError, match="Meep is not installed"):
        MeepAdapter.run_simulation(geometry, run_time=12.5)


def test_coupled_waveguide_analyzer_paths() -> None:
    analyzer = CrosstalkModel()
    assert analyzer.worst_case_isolation() == float("inf")

    pair = WaveguidePair(gap_nm=200.0, coupling_length_um=50.0)
    analyzer.add_pair(pair)

    transfer = analyzer.transfer_matrix(pair)
    assert transfer.shape == (2, 2)

    power_a, power_b = analyzer.compute_crosstalk(pair)
    assert power_a >= 0.0
    assert power_b >= 0.0
    assert analyzer.worst_case_isolation() == pytest.approx(pair.isolation_db)


def test_photonic_compiler_mzi_netlist_branch() -> None:
    """An MZI-typed target drives the MZI netlist branch (phase + amplitude
    directives), distinct from the microring coupling/detuning lines."""
    compiler = PhotonicCompiler(PhotonicTarget.lightmatter())  # modulator_type == "MZI"
    result = compiler.compile_bitstream(np.array([1, 0, 1], dtype=np.uint8))
    assert "ADD MZI mod_0" in result.netlist
    assert "mod_0:phase" in result.netlist
    assert "mod_0:amplitude" in result.netlist
    # The microring directives must be absent for an MZI target.
    assert "MICRORING" not in result.netlist


def test_generate_modulator_verilog_sources_parameterise_bit_width() -> None:
    """Both modulator SystemVerilog generators emit a module with the
    requested bus width threaded into the BW parameter."""
    compiler = PhotonicCompiler(PhotonicTarget.lightmatter())

    mzi_src = compiler.generate_mzi_verilog(bit_width=12)
    assert "module sc_photonic_mzi" in mzi_src
    assert "parameter BW = 12" in mzi_src

    microring_src = compiler.generate_microring_verilog(bit_width=10)
    assert "module sc_photonic_microring" in microring_src
    assert "parameter BW = 10" in microring_src


def test_fdtd2d_field_at_point_reads_ez_grid() -> None:
    """``field_at_point`` returns the scalar Ez value at a grid cell; an
    injected Gaussian source raises the central cell above the zero floor."""
    solver = FDTD2DSolver(nx=24, ny=16)
    assert solver.field_at_point(6, 8) == 0.0

    solver.inject_source(6, 8, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=3)
    centre = solver.field_at_point(6, 8)
    assert isinstance(centre, float)
    assert centre == pytest.approx(1.0)


def test_analyze_bank_single_waveguide_has_no_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bank of one waveguide forms zero adjacent and zero next-nearest
    pairs, so the pure-Python fallback collapses to the empty-bank defaults.

    The flag is forced off so the fallback runs regardless of whether the
    native crosstalk backend is built in this environment (the Rust path has
    its own return and would otherwise shadow the ``total == 0`` branch)."""
    monkeypatch.setattr("sc_neurocore.optics.photonic_emitter._HAS_RUST_PH", False)
    analyzer = CrosstalkModel()
    report = analyzer.analyze_bank(waveguides=1, gap_nm=200.0, coupling_length_um=50.0)
    assert report["num_pairs"] == 0
    assert report["num_near_pairs"] == 0
    assert report["num_far_pairs"] == 0
    assert report["worst_isolation_db"] == float("inf")
    assert report["mean_coupling_ratio"] == 0.0
    assert report["max_coupling_ratio"] == 0.0
    assert report["crosstalk_safe"] is True


def test_to_gdsii_activates_generic_pdk_when_none_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``get_active_pdk`` reports no active PDK, ``to_gdsii`` activates
    the generic PDK on demand rather than failing the export."""
    gf = pytest.importorskip(
        "gdsfactory",
        reason="gdsfactory is an optional dep (install via `pip install sc-neurocore[optics]`)",
    )

    real_get_active_pdk: Callable[[], object] = gf.get_active_pdk
    monkeypatch.setattr(gf, "gpdk", object())
    calls = {"n": 0}

    def _raise_first_then_real() -> object:
        # The export's own probe (first call) must hit the no-PDK branch; any
        # later resolution (e.g. gf.components.mzi defaults) sees the PDK the
        # fallback just activated.
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("no active PDK")
        return real_get_active_pdk()

    monkeypatch.setattr(gf, "get_active_pdk", _raise_first_then_real)

    result = CompilationResult(
        target="pdk_fallback_probe",
        num_modulators=2,
        optical_power_mean_mw=1.0,
        phase_coverage_rad=1.0,
        netlist="",
    )
    out_path = tmp_path / "pdk_fallback.gds"
    info = result.to_gdsii(str(out_path))

    assert calls["n"] >= 1
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert info["n_modulators"] == 2

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Branch tests for photonic emitter helpers

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.optics.photonic_emitter import (
    BitstreamToOptical,
    CompilationResult,
    CrosstalkModel,
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


def test_meep_adapter_build_and_mock_run() -> None:
    target = PhotonicTarget.lightmatter()
    geometry = MeepAdapter.build_waveguide_geometry(target, waveguide_width_um=0.6, length_um=12.0)
    assert geometry["cell_size"][0] == 12.0
    assert geometry["sources"][0]["type"] == "ContinuousSource"
    result = MeepAdapter.run_simulation({"wavelength_nm": 1310.0}, run_time=12.5)
    assert result["mock"] is True
    assert result["run_time"] == 12.5
    assert result["wavelength_nm"] == 1310.0


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

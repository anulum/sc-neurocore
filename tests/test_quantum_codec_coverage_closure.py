# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused quantum cognition and waveform codec coverage

"""Close terminal, quantum validation, MPS, and optional codec branches."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sc_neurocore.quantum_cognition import dashboard, gotm_brain
from sc_neurocore.quantum_cognition.kane_mapper import KaneSiliconMapper
from sc_neurocore.quantum_cognition.radical_pair import RadicalPairModel, RadicalPairParams
from sc_neurocore.quantum_cognition.spin_pool import SpinCouplingTensor, SpinPoolMPS
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec


def test_dashboard_helpers_cover_zero_and_colour_bands() -> None:
    """Terminal bars handle zero maxima and low, medium, and high bands."""

    assert "▁" in dashboard._heat_char(1.0, 0.0)
    assert dashboard._bar(1.0, 0.0, width=3) == "░░░"
    assert dashboard._YELLOW in dashboard._bar(0.5, 1.0)
    assert dashboard._RED in dashboard._bar(0.1, 1.0)
    assert dashboard._GREEN in dashboard._bar(0.9, 1.0)


def test_dashboard_draw_handles_terminal_failure_and_all_raster_bands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fallback-sized frame renders overflow, raster, and directive states."""

    def terminal_failure(_fallback: tuple[int, int]) -> tuple[int, int]:
        raise OSError("terminal unavailable")

    monkeypatch.setattr(dashboard.shutil, "get_terminal_size", terminal_failure)
    neurons = [SimpleNamespace(atp_level=0.5) for _ in range(30)]
    history = [
        {"n_spikes": 0, "directive": "FOCUS"},
        {"n_spikes": 1, "directive": "EXPLORE"},
        {"n_spikes": 4, "directive": "STABILIZE"},
        {"n_spikes": 9, "directive": "UNKNOWN"},
    ]
    brain = SimpleNamespace(
        pool=SimpleNamespace(entanglement_map=np.array([0.0, 1.0])),
        neurons=neurons,
        get_learning_state=lambda: {
            "n_neurons": len(neurons),
            "total_steps": 4,
            "total_spikes": 14,
            "bridge_backend": "numpy",
            "has_llm": False,
            "total_metabolic_failures": 0,
            "avg_entanglement": 0.5,
            "avg_atp": 0.5,
        },
        get_history=lambda: history,
    )
    view = dashboard.TerminalDashboard(max_raster_steps=4, clear_screen=True)

    view.draw(brain)
    rendered = capsys.readouterr().out

    assert "(+4 more)" in rendered
    assert "Spike Raster" in rendered
    assert "Directive History" in rendered
    assert repr(view) == "TerminalDashboard(max_raster=4)"


def test_local_llm_guidance_accepts_valid_directive_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local LLM responses are normalized, allowlisted, and exception-safe."""

    brain = object.__new__(gotm_brain.GOTMBrain)
    brain._llm_endpoint = object()
    monkeypatch.setattr(gotm_brain, "HAS_LLM", True)
    monkeypatch.setattr(gotm_brain, "_llm_chat", lambda _prompt, **_kwargs: "focus.")
    assert brain.get_llm_guidance("context") == "FOCUS"

    monkeypatch.setattr(gotm_brain, "_llm_chat", lambda _prompt, **_kwargs: "unsupported")
    assert brain.get_llm_guidance("context") == "STABILIZE"

    def fail_chat(_prompt: str, **_kwargs: object) -> str:
        raise RuntimeError("local endpoint failed")

    monkeypatch.setattr(gotm_brain, "_llm_chat", fail_chat)
    assert brain.get_llm_guidance("context") == "STABILIZE"


def test_kane_mapper_repr_exposes_physical_layout() -> None:
    """Mapper representation reports spacing, depth, and topology."""

    rendered = repr(KaneSiliconMapper())
    assert "spacing=" in rendered
    assert "depth=" in rendered
    assert "topology=" in rendered


def test_radical_pair_constructor_and_tensor_validation_boundaries() -> None:
    """Radical-pair models reject invalid quadrature, tensors, and bath size."""

    with pytest.raises(ValueError, match="quadrature_order"):
        RadicalPairModel(RadicalPairParams(quadrature_order=1))

    explicit = RadicalPairModel.from_hyperfine_tensors(
        tensors_1=[np.eye(3)],
        tensors_2=[],
        exchange_j=1.0,
        recombination_rate=0.1,
        lifetime_us=1.0,
        quadrature_order=2,
    )
    assert explicit.params.hyperfine_tensors_1

    malformed = RadicalPairModel(RadicalPairParams(hyperfine_tensors_1=[np.zeros((2, 2))]))
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        malformed._validated_tensors()

    rho, projector = RadicalPairModel._singlet_density_with_nuclear_bath(0)
    assert np.array_equal(rho, projector)

    oversized = RadicalPairModel(
        RadicalPairParams(hyperfine_tensors_1=[np.eye(3) for _ in range(9)])
    )
    with pytest.raises(ValueError, match="up to 8 nuclei"):
        oversized._hamiltonian(0.0)


def test_radical_pair_recombination_state_and_repr_boundaries() -> None:
    """Recombination inputs fail closed and state metadata remains inspectable."""

    with pytest.raises(ValueError, match="recombination_rate"):
        RadicalPairModel(RadicalPairParams(recombination_rate=0.0)).singlet_yield()
    with pytest.raises(ValueError, match="lifetime_us"):
        RadicalPairModel(RadicalPairParams(lifetime_us=0.0)).singlet_yield()

    model = RadicalPairModel()
    state = model.get_state()
    assert state["hyperfine_a_mhz"] == model.params.hyperfine_a
    assert "RadicalPairModel(" in repr(model)


def test_spin_pool_rejects_zero_norm_and_invalid_exact_couplings() -> None:
    """MPS export and exact evolution reject degenerate or malformed state."""

    pool = SpinPoolMPS(n_sites=2)
    pool.tensors = [
        np.zeros((1, 2, 1), dtype=np.complex128),
        np.zeros((1, 2, 1), dtype=np.complex128),
    ]
    with pytest.raises(ValueError, match="zero norm"):
        pool.to_statevector()

    pool = SpinPoolMPS(n_sites=2)
    with pytest.raises(IndexError, match="out of range"):
        pool.evolve_exact([SpinCouplingTensor(0, 2, np.eye(3))], 0.1)
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        pool.evolve_exact([SpinCouplingTensor(0, 1, np.zeros((2, 2)))], 0.1)
    with pytest.raises(IndexError, match="Two-site RDM"):
        pool._compute_rdm_two_site(1)


def test_spin_pool_heisenberg_handles_identity_and_reversed_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heisenberg routing skips identity pairs and normalizes reversed sites."""

    pool = SpinPoolMPS(n_sites=2)
    called: list[tuple[int, int, float]] = []
    monkeypatch.setattr(
        pool,
        "_apply_tebd_gate",
        lambda i, j, coupling: called.append((i, j, coupling)),
    )

    pool._apply_heisenberg_between(0, 0, 1.0)
    pool._apply_heisenberg_between(1, 0, 2.0)

    assert called == [(0, 1, 2.0)]


class _ZstdCompressor:
    """Minimal zstandard compressor double preserving a visible payload."""

    def __init__(self, *, level: int) -> None:
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return b"zstd" + data


def test_waveform_codec_uses_optional_zstd_and_wavelet_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed optional compressors and wavelets execute their maintained path."""

    zstandard = ModuleType("zstandard")
    zstandard.ZstdCompressor = _ZstdCompressor  # type: ignore[attr-defined]
    pywt = ModuleType("pywt")
    pywt.wavedec = lambda data, _wavelet, axis: [data, data * 0.0]  # type: ignore[attr-defined]
    pywt.threshold = lambda data, _threshold, mode: data  # type: ignore[attr-defined]
    pywt.waverec = lambda coeffs, _wavelet, axis: coeffs[0]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zstandard", zstandard)
    monkeypatch.setitem(sys.modules, "pywt", pywt)
    codec = WaveformCodec(snippet_samples=2)

    snippet = codec._compress_snippets(
        [np.array([1.0, -1.0])],
        [0],
        [np.array([0.25, -0.25])],
    )
    background = codec._compress_background(np.arange(32, dtype=np.float64).reshape(16, 2))

    assert b"zstd" in snippet
    assert b"zstd" in background

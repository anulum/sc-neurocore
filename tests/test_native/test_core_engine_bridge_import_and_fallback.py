# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (import_and_fallback) from former test_core_engine_bridge.py

from __future__ import annotations

from core_engine_bridge_support import *  # noqa: F403


def test_import_time_oserror_keeps_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the bridge importable when the shared object exists but cannot load."""
    real_cdll = ct.CDLL

    def failing_cdll(name: Any, *args: Any, **kwargs: Any) -> ct.CDLL:
        name_as_text = name.decode() if isinstance(name, bytes) else str(name)
        if Path(name_as_text).name == "libcore_engine.so":
            raise OSError("synthetic loader failure")
        return real_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ct, "CDLL", failing_cdll)

    namespace = runpy.run_path(
        str(Path(ceb.__file__).resolve()),
        run_name="_sc_neurocore_core_engine_bridge_loader_failure",
    )

    assert namespace["_HAS_CORE_ENGINE"] is False
    assert namespace["_lib"] is None


def test_get_lib_raises_when_unloaded() -> None:
    """Reject native dispatch when no core engine library is loaded."""
    ceb._lib = None
    with pytest.raises(RuntimeError, match="not loaded"):
        ceb._get_lib()


def test_scalar_fallbacks_when_native_absent() -> None:
    """Exercise scalar and list fallbacks when the native engine is unavailable."""
    ceb._HAS_CORE_ENGINE = False
    assert ceb.is_available() is False
    assert ceb.sc_multiply(0b1010, 0b1100) == 0b1000
    assert ceb.sc_mux(0xAAAAAAAA, 0x55555555, 0xFFFFFFFF) == 0xAAAAAAAA
    assert ceb.sc_popcount(0b101101) == 4
    assert ceb.sc_popcount64((1 << 63) | 0b101) == 3
    assert ceb.sc_popcount_packed([0b1010, 0b1111]) == 6
    assert ceb.sc_scc_packed([0xAAAAAAAAAAAAAAAA], [0xAAAAAAAAAAAAAAAA]) == pytest.approx(1.0)
    assert ceb.sc_scc_packed([0xAAAAAAAAAAAAAAAA], [0x5555555555555555]) == pytest.approx(-1.0)
    assert ceb.sc_scc_packed([0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]) == pytest.approx(0.0)


def test_python_scc_fallback_honors_logical_bit_length() -> None:
    """Mask packed SCC fallbacks to the caller's logical bit length."""
    ceb._HAS_CORE_ENGINE = False

    assert ceb.sc_scc_packed([0b1010_1010], [0b1010_1010], bit_length=8) == pytest.approx(1.0)
    assert ceb.sc_scc_packed([0b1010_1010], [0b0101_0101], bit_length=8) == pytest.approx(-1.0)

    masked = ceb.sc_scc_packed([0xFFFF_FFFF_FFFF_FF0F], [0xFFFF_FFFF_FFFF_FF0F], bit_length=8)
    assert masked == pytest.approx(1.0)


def test_numpy_scc_empty_input_returns_zero() -> None:
    """Return zero for empty NumPy SCC inputs before native dispatch."""
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = cast(ct.CDLL, _FakeCoreLib())
    assert ceb.sc_scc_packed_np(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64)) == 0.0


def test_lfsr_and_scc_python_fallbacks_when_engine_absent() -> None:
    """Compose Python LFSR and SCC fallbacks when the native engine is absent."""
    # The autouse fixture restores _HAS_CORE_ENGINE afterwards.
    ceb._HAS_CORE_ENGINE = False
    packed = ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=128)
    assert packed.dtype == np.uint64
    correlation = ceb.sc_scc_packed_np(np.asarray(packed), np.asarray(packed), bit_length=128)
    assert -1.0 <= correlation <= 1.0


def test_python_lfsr_fallback_escapes_zero_state() -> None:
    """Preserve progress if a direct helper call starts from the invalid zero state."""
    words = ceb._python_lfsr_encode_packed(seed=0, threshold=1, bit_length=2)
    assert words == [1]

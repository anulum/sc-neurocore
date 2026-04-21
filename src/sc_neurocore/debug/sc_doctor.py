# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive SC Doctor (ported from dynamic_adaptation/lib.rs)

"""Runtime dynamic SC adaptation with Hamming(7,4) ECC.

Monitors correlation metrics and auto-tunes bitstream length.
Enables ECC when length exceeds threshold to protect against noise.

The three hot methods (:meth:`ScDoctor.adapt`, :meth:`ScDoctor.encode_ecc`,
:meth:`ScDoctor.decode_ecc`) dispatch to the Rust ``stochastic_doctor_core``
PyO3 extension when the compiled `.so` is importable. When it is not, a
bit-exact pure-Python fallback is used so behaviour is identical.
"""

from __future__ import annotations

try:
    from sc_neurocore.stochastic_doctor import stochastic_doctor_core as _sdc

    # `stochastic_doctor/__init__.py` now returns `None` on missing .so
    # rather than raising ImportError, so module import never hard-fails.
    # The real gate is whether the attribute is a loaded extension.
    _HAS_RUST_DOCTOR = _sdc is not None
except ImportError:
    _sdc = None
    _HAS_RUST_DOCTOR = False


class ScDoctor:
    """Adaptive bitstream length controller with optional ECC.

    Correlation-driven feedback loop:
    - High correlation (>0.15): double bitstream length
    - Low correlation (<0.05): halve bitstream length (floor 256)
    - ECC auto-enabled when length exceeds 2048
    """

    def __init__(self, initial_length: int = 256, target_precision: float = 0.95):
        self.current_bitstream_length = initial_length
        self.target_precision = target_precision
        self.error_correction_enabled = False

    def adapt(self, current_correlation: float, popcount: int = 0) -> None:
        """Analyze correlation and adjust bitstream length.

        Dispatches to the Rust ``stochastic_doctor_core.py_sc_doctor_adapt``
        when available; falls back to the bit-exact Python implementation.

        Parameters
        ----------
        current_correlation : float
            Current SC correlation metric (SCC estimate).
        popcount : int
            Current popcount (reserved for future use).
        """
        if _HAS_RUST_DOCTOR:
            new_length, ecc_enabled = _sdc.py_sc_doctor_adapt(
                int(self.current_bitstream_length),
                bool(self.error_correction_enabled),
                float(current_correlation),
            )
            self.current_bitstream_length = int(new_length)
            self.error_correction_enabled = bool(ecc_enabled)
            return

        if current_correlation > 0.15:
            self.current_bitstream_length *= 2
            if self.current_bitstream_length > 2048:
                self.error_correction_enabled = True
        elif current_correlation < 0.05 and self.current_bitstream_length > 256:
            self.current_bitstream_length //= 2
            self.error_correction_enabled = False

    def encode_ecc(self, data: int) -> int:
        """Hamming(7,4) encode a 4-bit chunk → 7-bit codeword.

        If ECC is disabled, returns lower 4 bits unchanged. Hot path
        dispatches to ``stochastic_doctor_core.py_hamming74_encode``.
        """
        if not self.error_correction_enabled:
            return data & 0x0F

        if _HAS_RUST_DOCTOR:
            return int(_sdc.py_hamming74_encode(int(data)))

        d1 = (data >> 3) & 1
        d2 = (data >> 2) & 1
        d3 = (data >> 1) & 1
        d4 = data & 1

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4

        return (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4

    def decode_ecc(self, encoded: int) -> int:
        """Hamming(7,4) decode with single-bit error correction.

        If ECC is disabled, returns lower 4 bits unchanged. Hot path
        dispatches to ``stochastic_doctor_core.py_hamming74_decode``.
        """
        if not self.error_correction_enabled:
            return encoded & 0x0F

        if _HAS_RUST_DOCTOR:
            return int(_sdc.py_hamming74_decode(int(encoded)))

        p1 = (encoded >> 6) & 1
        p2 = (encoded >> 5) & 1
        d1 = (encoded >> 4) & 1
        p3 = (encoded >> 3) & 1
        d2 = (encoded >> 2) & 1
        d3 = (encoded >> 1) & 1
        d4 = encoded & 1

        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4

        syndrome = (s3 << 2) | (s2 << 1) | s1

        corrected = encoded
        bit_positions = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}
        if syndrome in bit_positions:
            corrected ^= 1 << bit_positions[syndrome]

        cd1 = (corrected >> 4) & 1
        cd2 = (corrected >> 2) & 1
        cd3 = (corrected >> 1) & 1
        cd4 = corrected & 1

        return (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4

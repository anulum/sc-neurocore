# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_quantizer.py

"""Module-level tests from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

def test_block_floating_benchmark_contract_matches_rust_envelope() -> None:
    """The documented Python/Rust benchmark workload must share one envelope."""
    n_inputs = 64
    n_outputs = 32
    weights = np.array(
        [((idx * 23 + 3) % 1025 - 512) / 512.0 for idx in range(n_inputs * n_outputs)],
        dtype=np.float64,
    ).reshape(n_outputs, n_inputs)
    inputs = np.array(
        [((idx * 19 + 5) % 257 - 128) / 256.0 for idx in range(n_inputs)],
        dtype=np.float64,
    )

    compiled = compile_dense_block_floating(weights, fmt="BFP16E3X32")
    envelope = compiled.precision_envelope_report(inputs)

    assert int(np.sum(compiled.mantissas.astype(np.int64))) == -15
    assert int(np.sum(compiled.exponents.astype(np.int64))) == 0
    assert envelope.max_abs_bound_code == 610_816
    assert envelope.conservative_overflow_free

    probe = compile_dense_block_floating(
        np.full((n_outputs, n_inputs), 8192.0, dtype=np.float64),
        fmt="BFP16E3X32",
    )
    probe_envelope = probe.precision_envelope_report(np.full(n_inputs, 32767.0, dtype=np.float64))

    assert int(np.sum(probe.mantissas.astype(np.int64))) == 33_554_432
    assert int(np.sum(probe.exponents.astype(np.int64))) == 128
    assert probe_envelope.max_abs_bound_code == 1_125_865_547_104_256
    assert not probe_envelope.conservative_overflow_free
def test_mixed_dense_benchmark_contract_matches_rust_envelope() -> None:
    """Canonical Q8.8/Q16.16 benchmark contract matches the Rust envelope."""

    from sc_neurocore.compiler.quantizer import (
        QFormatMixed,
        compile_dense_mixed_precision,
    )

    n_inputs = 64
    n_outputs = 32
    weights = np.array(
        [((idx * 17 + 11) % 513 - 256) / 256.0 for idx in range(n_inputs * n_outputs)],
        dtype=np.float64,
    ).reshape(n_outputs, n_inputs)
    inputs = np.array(
        [((idx * 19 + 5) % 257 - 128) / 256.0 for idx in range(n_inputs)],
        dtype=np.float64,
    )

    mixed_format = QFormatMixed(scale_per_tensor=False)
    compiled = compile_dense_mixed_precision(weights, fmt=mixed_format)
    safe_envelope = compiled.precision_envelope_report(inputs)
    _, safe_overflow = compiled.forward_with_overflow(inputs)

    assert compiled.tensor_scale == 1.0
    assert int(safe_envelope.max_abs_bound_code) == 531_400
    assert safe_envelope.conservative_overflow_free
    assert int(safe_envelope.min_headroom_code) == 2_146_952_247
    assert int(safe_envelope.required_total_bits) == 21
    assert int(safe_envelope.required_integer_bits) == 5
    assert int(safe_envelope.width_headroom_bits) == 11
    assert int(np.count_nonzero(safe_overflow)) == 0

    probe = compile_dense_mixed_precision(
        np.full((n_outputs, n_inputs), 127.0, dtype=np.float64),
        fmt=mixed_format,
    )
    probe_inputs = np.full(n_inputs, 32767.0, dtype=np.float64)
    probe_envelope = probe.precision_envelope_report(probe_inputs)
    _, probe_overflow = probe.forward_with_overflow(probe_inputs)

    assert int(probe_envelope.max_abs_bound_code) == 17_454_214_414_336
    assert not probe_envelope.conservative_overflow_free
    assert int(probe_envelope.required_total_bits) == 45
    assert int(probe_envelope.required_integer_bits) == 29
    assert int(probe_envelope.width_headroom_bits) == -13
    assert probe_envelope.saturation_required
    assert int(np.count_nonzero(probe_overflow)) == n_outputs

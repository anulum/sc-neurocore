# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive codec compatibility and timing

"""Exercise predictor state and quantised reconstruction through public packets."""

import numpy as np
import pytest

from sc_neurocore.spike_codec import PredictiveSpikeCodec, SpikeCodec


@pytest.mark.parametrize(("threshold", "value"), [(0.1, 0), (0.9, 1)])
def test_world_model_threshold_travels_with_packet(threshold: float, value: int) -> None:
    """Decode a nondefault threshold without the encoding instance or its state."""
    spikes = np.full((4, 2), value, dtype=np.int8)
    encoder = PredictiveSpikeCodec(predictor="world_model", threshold=threshold)
    packet, result = encoder.compress(spikes)

    assert result.lossless
    assert packet.startswith(b"PSCT")
    np.testing.assert_array_equal(PredictiveSpikeCodec().decompress(packet, 4, 2), spikes)
    with pytest.raises(ValueError, match="Truncated predictive header"):
        PredictiveSpikeCodec().decompress(packet[:15], 4, 2)


def test_default_world_model_packet_remains_compatible() -> None:
    """Read an original default-threshold packet and reproduce its exact bytes."""
    original = bytes.fromhex(
        "50534357083f747ae147ae147bace1010000000400000002000000020000000000000000"
    )
    expected = np.zeros((4, 2), dtype=np.int8)
    np.testing.assert_array_equal(PredictiveSpikeCodec().decompress(original, 4, 2), expected)

    packet, _ = PredictiveSpikeCodec(predictor="world_model").compress(expected)
    assert packet == original


@pytest.mark.parametrize(
    ("predictor", "packet_hex"),
    [
        (
            "ema",
            "505343583fe199999999999a3fe0000000000000010000001000000003000000030c0608"
            "00000003000300000400000001010000000202000000030300000004030000002691117615e8",
        ),
        (
            "lfsr",
            "5053434c00c0000101000000100000000300000003090409000000030003000005000000"
            "0101000000020200000003030000000404000000050400000024b149fd30c0",
        ),
        (
            "context",
            "505343430201000000100000000300000003080508000000030003000005000000010100"
            "00000202000000030300000004040000000504000000225cacf69400",
        ),
        (
            "world_model",
            "50534357033fa15c56760bbef0002a0100000010000000030000000309040b00000003000300"
            "00050000000101000000020200000003030000000404000000050400000027b494fe2198",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["python", "auto"])
def test_legacy_predictive_packets_keep_nontrivial_residuals(
    predictor: str, packet_hex: str, backend: str
) -> None:
    """Read a pre-upgrade packet and reproduce its bytes after predictor state changes."""
    rows = (
        "101",
        "100",
        "001",
        "111",
        "110",
        "000",
        "101",
        "001",
        "001",
        "111",
        "000",
        "100",
        "101",
        "011",
        "100",
        "000",
    )
    spikes = np.array([[int(bit) for bit in row] for row in rows], dtype=np.int8)
    original = bytes.fromhex(packet_hex)
    decoder = PredictiveSpikeCodec(backend=backend)
    np.testing.assert_array_equal(decoder.decompress(original, *spikes.shape), spikes)

    # At the stored learning rate, a learned world-model probability is 1e-8 above 0.5.
    alpha = {"ema": 0.55, "world_model": 0.03390760603894971}.get(predictor, 0.005)
    encoder = PredictiveSpikeCodec(
        predictor=predictor,
        alpha=alpha,
        alpha_q8=192,
        context_bits=2 if predictor == "context" else 3,
        seed=1 if predictor == "lfsr" else 42,
        backend=backend,
    )
    packet, result = encoder.compress(spikes)
    assert result.lossless
    assert packet == original


def test_quantised_predictive_encoder_refuses_rows_outside_the_wire_field() -> None:
    """A zero-channel raster can exceed the wire row limit without allocating samples."""
    spikes = np.empty((1 << 32, 0), dtype=np.int8)
    encoder = PredictiveSpikeCodec(base_mode="lossy", timing_precision=2, backend="python")
    with pytest.raises(ValueError, match="dimensions must fit uint32"):
        encoder.compress(spikes)


@pytest.mark.parametrize("predictor", ["ema", "lfsr", "context", "world_model"])
@pytest.mark.parametrize("backend", ["python", "auto"])
def test_lossy_prediction_reconstructs_the_same_bins_as_isi(predictor: str, backend: str) -> None:
    """Merge complete bins before feedback and discard spikes in the final partial bin."""
    spikes = np.array([[0, 1], [1, 0], [0, 0], [0, 1], [0, 0], [0, 0], [1, 1]], dtype=np.int8)
    baseline = SpikeCodec(mode="lossy", timing_precision=2)
    isi_packet, _ = baseline.compress(spikes)
    expected = baseline.decompress(isi_packet, 7, 2)
    encoder = PredictiveSpikeCodec(
        predictor=predictor,
        alpha=0.5,
        threshold=0.25,
        base_mode="lossy",
        timing_precision=2,
        backend=backend,
    )
    packet, result = encoder.compress(spikes)
    decoder = PredictiveSpikeCodec(backend=backend)

    assert packet.startswith(b"PSCQ")
    assert not result.lossless
    np.testing.assert_array_equal(decoder.decompress(packet, 7, 2), expected)
    np.testing.assert_array_equal(decoder.decompress(packet, 3, 2), expected[:3])
    with pytest.raises(ValueError, match="incompatible row count"):
        decoder.decompress(packet, 6, 2)


def test_quantised_packet_refuses_truncated_or_inconsistent_timing_header() -> None:
    """Reject malformed timing data before allocating or predicting a raster."""
    encoder = PredictiveSpikeCodec(base_mode="lossy", timing_precision=2)
    packet, _ = encoder.compress(np.zeros((4, 1), dtype=np.int8))
    decoder = PredictiveSpikeCodec()

    with pytest.raises(ValueError, match="Truncated quantised predictive header"):
        decoder.decompress(packet[:8], 4, 1)
    with pytest.raises(ValueError, match="Invalid predictive timing precision"):
        decoder.decompress(packet[:8] + (1).to_bytes(8, "big") + packet[16:], 4, 1)


def test_ema_quantisation_does_not_invent_an_impulse() -> None:
    """Recover one quantised event when predictor feedback would otherwise add one."""
    spikes = np.array([[1], [0]], dtype=np.int8)
    encoder = PredictiveSpikeCodec(alpha=0.5, threshold=0.25, base_mode="lossy", timing_precision=2)
    packet, _ = encoder.compress(spikes)
    np.testing.assert_array_equal(PredictiveSpikeCodec().decompress(packet, 2, 1), spikes)


@pytest.mark.parametrize("predictor", ["ema", "lfsr", "context", "world_model"])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_predictive_packet_keeps_empty_raster_geometry(
    predictor: str, shape: tuple[int, int]
) -> None:
    """Round-trip empty time and empty channel axes without native division by zero."""
    spikes = np.zeros(shape, dtype=np.int8)
    encoder = PredictiveSpikeCodec(predictor=predictor)
    packet, _ = encoder.compress(spikes)
    np.testing.assert_array_equal(PredictiveSpikeCodec().decompress(packet, *shape), spikes)


@pytest.mark.parametrize(
    "spikes",
    [
        np.array([0, 1], dtype=np.int8),
        np.array([[0, 256]], dtype=np.int64),
        np.array([[1.0, 0.5]], dtype=np.float64),
        np.array([["1", "0"]]),
    ],
)
def test_predictive_encoder_rejects_invalid_samples_before_narrowing(
    spikes: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    """Refuse invalid samples and rank before predictor state or a packet is created."""
    with pytest.raises(ValueError, match="binary raster|samples must be 0 or 1"):
        PredictiveSpikeCodec().compress(spikes)


@pytest.mark.parametrize(("predictor", "context_bits"), [("context", 64), ("world_model", 0)])
def test_predictive_history_parameter_is_checked_on_write_and_read(
    predictor: str, context_bits: int
) -> None:
    """Refuse a history the predictor cannot represent on both packet paths."""
    spikes = np.zeros((4, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="history length"):
        PredictiveSpikeCodec(predictor=predictor, context_bits=context_bits).compress(spikes)

    packet, _ = PredictiveSpikeCodec(predictor=predictor).compress(spikes)
    invalid = packet[:4] + bytes([context_bits]) + packet[5:]
    with pytest.raises(ValueError, match="history length"):
        PredictiveSpikeCodec().decompress(invalid, *spikes.shape)


def test_predictive_encoder_rejects_invalid_predictor_parameters() -> None:
    """Refuse unknown predictors, invalid seeds and non-finite predictor coefficients."""
    spikes = np.zeros((4, 2), dtype=np.int8)

    with pytest.raises(ValueError, match="Invalid predictor"):
        PredictiveSpikeCodec(predictor="unknown").compress(spikes)
    with pytest.raises(ValueError, match="LFSR seed"):
        PredictiveSpikeCodec(predictor="lfsr", seed=0).compress(spikes)
    with pytest.raises(ValueError, match="finite"):
        PredictiveSpikeCodec(alpha=float("nan")).compress(spikes)

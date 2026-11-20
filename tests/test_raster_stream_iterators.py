# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Incremental spike-raster frame consumer tests

"""Exercise bounded pull, cancellation and recovery through the frame API."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.spike_codec import StreamingSpikeCodec
from tests.engine_requirement import require_engine


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_pull_stream_conserves_rows_across_arbitrary_chunks(backend: str) -> None:
    """Pull only enough chunks for one frame and conserve the short tail."""
    if backend == "rust":
        require_engine()
    raster = (np.arange(39).reshape(13, 3) % 5 == 0).astype(np.int8)
    codec = StreamingSpikeCodec(4, backend=backend)
    pulls: list[int] = []

    def chunks() -> Iterator[NDArray[np.int8]]:
        for chunk in (raster[:3], raster[3:9], raster[9:]):
            pulls.append(len(chunk))
            yield chunk

    encoder = codec.iter_compress(chunks())
    assert pulls == []
    first = next(encoder)
    assert pulls == [3, 6]
    decoded = codec.iter_decompress(chain((first,), encoder))
    first_window = next(decoded)
    assert pulls == [3, 6]
    recovered = [first_window, *decoded]
    assert [part.shape[0] for part in recovered] == [4, 4, 4, 1]
    np.testing.assert_array_equal(np.concatenate(recovered), raster)
    assert list(codec.iter_compress([raster[:0]])) == []
    assert list(codec.iter_decompress([])) == []


def test_cancel_stops_input_and_new_iterator_resets_state() -> None:
    """Close an in-flight encoder and decoder without consuming another item."""
    codec = StreamingSpikeCodec(3, backend="python")
    raster = np.arange(18).reshape(6, 3) % 2

    def chunks() -> Iterator[NDArray[np.int64]]:
        yield raster[:4]
        pytest.fail("closed encoder requested another chunk")

    encoder = codec.iter_compress(chunks())
    first = next(encoder)
    encoder.close()
    assert list(encoder) == []
    restarted = list(codec.iter_compress([raster[4:]]))
    assert len(restarted) == 1
    np.testing.assert_array_equal(codec.decompress_frame(restarted[0]), raster[4:])

    def frames() -> Iterator[bytes]:
        yield first
        pytest.fail("closed decoder requested another frame")

    decoder = codec.iter_decompress(frames())
    np.testing.assert_array_equal(next(decoder), raster[:3])
    decoder.close()
    assert list(decoder) == []


def test_stream_rejects_geometry_changes_and_recovers_at_a_frame_boundary() -> None:
    """Leave earlier frames usable after a damaged, oversized or inconsistent frame."""
    codec = StreamingSpikeCodec(3, backend="python")
    source = np.ones((3, 2), dtype=np.int8)
    short = codec.compress_frame(source[:2])
    full = codec.compress_frame(source)
    groups = codec.iter_compress([source, np.ones((2, 3), dtype=np.int8)])
    assert next(groups) == full
    with pytest.raises(ValueError, match="channel count changed"):
        next(groups)

    broken = codec.iter_decompress([full, full[:-1], full])
    np.testing.assert_array_equal(next(broken), source)
    with pytest.raises(ValueError, match="truncated raster frame mask"):
        next(broken)
    assert list(broken) == []
    np.testing.assert_array_equal(next(codec.iter_decompress([full])), source)

    with pytest.raises(ValueError, match="followed a short final frame"):
        list(codec.iter_decompress([short, full]))
    with pytest.raises(ValueError, match="frame dimensions changed"):
        list(codec.iter_decompress([full, codec.compress_frame(source[:, :1])]))
    with pytest.raises(ValueError, match="frame dimensions changed"):
        list(StreamingSpikeCodec(2, backend="python").iter_decompress([full]))

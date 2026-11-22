# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike codecs on the Python path and under an engine without their codec

"""Hold the spike codecs' Python refusals and their behaviour under an older engine.

Where the engine provides a codec, ``backend="auto"`` encodes and decodes natively, so the
Python packet parsers and limits run only when ``backend="python"`` is asked for. These tests
ask for it, so the Python refusals are held on every host.

An engine build that lacks a codec (an older engine, or one built without that codec) is a
real installation. In a child process, the engine package is one whose extension lacks the
codec symbols, installed before the codecs are imported. There ``backend="rust"`` must refuse
with the reason, and ``backend="auto"`` must fall back to Python and still round-trip.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sc_neurocore.spike_codec import AERSpikeCodec, DeltaSpikeCodec, SpikeCodec, probe_codec
from sc_neurocore.spike_codec.wire_fields import uint32_field


@pytest.mark.parametrize("value", [-1, 1 << 32])
def test_a_uint32_field_refuses_a_value_outside_its_range_by_name(value: int) -> None:
    """A header field refuses a negative value and one of 2**32 or more, naming the field.

    Parameters
    ----------
    value : int
        A value outside ``[0, 2**32)``.
    """
    with pytest.raises(ValueError, match=r"^Huffman bit count must fit uint32$"):
        uint32_field(value, "Huffman bit count")


def test_a_uint32_field_accepts_its_whole_range() -> None:
    """Both ends of ``[0, 2**32)`` are accepted unchanged, as plain ``int``."""
    assert uint32_field(0, "Huffman bit count") == 0
    assert uint32_field(np.uint32((1 << 32) - 1), "Huffman bit count") == (1 << 32) - 1


def test_isi_dimensions_keep_their_refusal_message() -> None:
    """The ISI codec still refuses a dimension of 2**32 with its own message."""
    with pytest.raises(ValueError, match=r"^ISI dimensions must fit uint32$"):
        SpikeCodec(backend="python").compress(np.zeros((1 << 32, 0), dtype=np.int8))


def test_python_address_event_decoder_refuses_a_foreign_header() -> None:
    """The Python AER parser refuses a packet whose magic is neither AERX nor AERI."""
    with pytest.raises(ValueError, match="Invalid header magic for AER"):
        AERSpikeCodec(backend="python").decompress(b"XXXX" + b"\x00" * 100)


def test_python_delta_decoder_refuses_a_foreign_header() -> None:
    """The Python DSCX parser refuses a packet whose magic is not DSCX."""
    with pytest.raises(ValueError, match="Invalid header magic for DSCX"):
        DeltaSpikeCodec(backend="python").decompress(b"XXXX" + b"\x00" * 100, 10, 5)


def test_python_varint_packet_refuses_the_ambiguous_raster_length() -> None:
    """The Python ISI encoder refuses T = 2**24 in varint form, as the native one does."""
    raster = np.zeros((2**24, 1), dtype=np.int8)
    raster[-1, 0] = 1
    with pytest.raises(ValueError, match="Huffman marker collision"):
        SpikeCodec(entropy="varint", backend="python").compress(raster)


#: The real engine's ISI codec functions, taken before the stand-in replaces the engine.
_TAKE_THE_ISI_CODEC = """
import importlib, sys
real = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
SYMBOLS = {name: getattr(real, name) for name in ("isi_encode_raster", "isi_decode_raster")}
for key in [key for key in sys.modules if key.split(".")[0] == "sc_neurocore_engine"]:
    del sys.modules[key]
"""

#: An engine package whose extension carries only ``SYMBOLS``, installed before any codec import.
_INSTALL_THE_STAND_IN = """
import sys, types
package = types.ModuleType("sc_neurocore_engine")
package.__path__ = []
extension = types.ModuleType("sc_neurocore_engine.sc_neurocore_engine")
for name, value in SYMBOLS.items():
    setattr(extension, name, value)
package.sc_neurocore_engine = extension
sys.modules["sc_neurocore_engine"] = package
sys.modules["sc_neurocore_engine.sc_neurocore_engine"] = extension
"""


def _run_child(body: str, *, keep_isi: bool) -> None:
    """Run ``body`` in a child whose engine lacks the codecs, keeping the ISI codec if asked.

    Parameters
    ----------
    body : str
        The program to run after the stand-in engine is installed.
    keep_isi : bool
        Whether the stand-in carries the real engine's ISI codec functions.
    """
    take = _TAKE_THE_ISI_CODEC if keep_isi else "SYMBOLS = {}\n"
    program = textwrap.dedent(take) + textwrap.dedent(_INSTALL_THE_STAND_IN) + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    ("codec", "reason"),
    [
        ("SpikeCodec()", "the ISI codec"),
        ("AERSpikeCodec()", "the address-event codec"),
        ("DeltaSpikeCodec()", "the DSCX codec"),
        ("StreamingSpikeCodec()", "the raster stream codec"),
    ],
)
def test_an_engine_without_the_codec_refuses_rust_and_falls_back_to_python(
    codec: str, reason: str
) -> None:
    """Under an engine that lacks the codec, rust refuses with the reason and auto round-trips.

    Parameters
    ----------
    codec : str
        The codec's constructor call.
    reason : str
        The codec the refusal must name.
    """
    rust = codec.replace("()", "(backend='rust')")
    _run_child(
        f"""
        import numpy as np
        from sc_neurocore.spike_codec import (
            AERSpikeCodec, DeltaSpikeCodec, SpikeCodec, StreamingSpikeCodec,
        )
        try:
            {rust}
        except ValueError as refusal:
            assert "backend 'rust' needs sc_neurocore_engine with {reason}" in str(refusal), refusal
        else:
            raise AssertionError("backend='rust' was not refused")
        codec = {codec}
        assert codec.backend == "python", codec.backend
        spikes = (np.random.default_rng(7).random((64, 6)) < 0.1).astype(np.int8)
        packet, _ = codec.compress(spikes)
        if isinstance(codec, AERSpikeCodec):
            restored = codec.decompress(packet)
        else:
            restored = codec.decompress(packet, *spikes.shape)
        assert np.array_equal(restored, spikes)
        """,
        keep_isi=False,
    )


def test_an_engine_with_only_the_isi_codec_refuses_rust_prediction() -> None:
    """With the ISI codec but no predictor, rust prediction is refused and auto predicts in Python."""
    if "rust" not in probe_codec("isi").backend_choices:
        # Without the real ISI codec there is no engine to keep it from; the refusal
        # above already holds the engine-less case.
        assert probe_codec("isi").backend_choices == ("auto", "python")
        return
    _run_child(
        """
        from sc_neurocore.spike_codec import PredictiveSpikeCodec
        codec = PredictiveSpikeCodec(predictor="ema", backend="rust")
        try:
            codec.predictor_backend
        except ValueError as refusal:
            assert "backend 'rust' needs sc_neurocore_engine with ema prediction" in str(refusal)
        else:
            raise AssertionError("rust prediction was not refused")
        assert PredictiveSpikeCodec(predictor="ema").predictor_backend == "python"
        """,
        keep_isi=True,
    )

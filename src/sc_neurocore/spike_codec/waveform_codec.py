# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Raw waveform codec: spike detection + compression pipeline

"""End-to-end neural waveform compression.

Full pipeline from raw 10-bit ADC samples to compressed bytes:
    1. Threshold-crossing spike detection
    2. Spike timing → binary raster → ISI spike codec
    3. Spike waveform snippets → template library + per-snippet scale + residuals
    4. Background signal → downsampling + closed-loop delta quantisation

The ``spike``, ``waveform`` and ``full`` modes are lossy with respect to the
raw signal: they keep spike timing exactly and approximate the rest. The
``lossless`` mode reconstructs every input sample exactly: per channel it
maps the samples onto their ADC lattice (offset and step), applies the best
fixed polynomial predictor of order 0-3 and codes the residuals with
block-adaptive Rice codes. ``WaveformCodec.decompress`` decodes every mode.

Operates on (T, N) int16 arrays (10-bit ADC values in int16 container).
The heavy kernels (detection, template matching, background quantisation,
lossless coding) run in Rust when ``sc_neurocore_engine`` provides them and
otherwise in NumPy; both backends write byte-identical streams.
Wire format ``WFC2`` ends with a CRC-32 of everything before it, so a
corrupted stream is refused instead of decoded. The earlier ``WFCX`` streams
carry no checksum and lack per-snippet scales and the background's first row,
so only their spike raster is decoded.
"""

from __future__ import annotations

import importlib
import math
import struct
import zlib
from dataclasses import dataclass
from math import isfinite
from types import ModuleType
from typing import Any

import numpy as np

from .codec import SpikeCodec

WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES = 1
WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES = 255
WAVEFORM_CODEC_MIN_TEMPLATES = 1
WAVEFORM_CODEC_MAX_HEADER_COUNT = 65535
WAVEFORM_CODEC_MAX_TEMPLATES = WAVEFORM_CODEC_MAX_HEADER_COUNT
WAVEFORM_CODEC_MIN_QUANTIZE_BITS = 1
WAVEFORM_CODEC_MAX_QUANTIZE_BITS = 8
WAVEFORM_CODEC_VALID_MODES = ("full", "waveform", "spike", "lossless")
WAVEFORM_CODEC_MODE_BYTES = {"full": 0, "waveform": 1, "spike": 2, "lossless": 3}
WAVEFORM_CODEC_MODES_BY_BYTE = {value: key for key, value in WAVEFORM_CODEC_MODE_BYTES.items()}
# Background downsampling factor of the ``full`` mode.
WAVEFORM_CODEC_BACKGROUND_DOWNSAMPLE = 16
# Lossless mode: residuals per Rice block, largest Rice parameter, and the
# unary length that marks an escaped (verbatim 32-bit) residual.
WAVEFORM_CODEC_LOSSLESS_BLOCK = 256
WAVEFORM_CODEC_LOSSLESS_MAX_RICE_K = 24
WAVEFORM_CODEC_LOSSLESS_ESCAPE = 32
WAVEFORM_CODEC_LOSSLESS_MAX_ORDER = 3
# Decoding refuses streams declaring more samples than this unless the caller
# raises the limit (a corrupt header must not allocate tens of gigabytes).
WAVEFORM_CODEC_MAX_DECODE_SAMPLES = 1 << 28
_HEADER_FORMAT = "!IIHHBBBB"
# Kernel backends: "rust" (sc_neurocore_engine), "python" (NumPy reference),
# "auto" (Rust when the installed engine provides every kernel). Both write
# byte-identical streams.
WAVEFORM_CODEC_BACKENDS = ("auto", "python", "rust")
_NATIVE_KERNELS = (
    "waveform_codec_noise_thresholds",
    "waveform_codec_extract_background",
    "waveform_codec_detect_spikes",
    "waveform_codec_template_match",
    "waveform_codec_quantize_background",
    "waveform_codec_reconstruct_background",
    "waveform_codec_lossless_encode_channel",
    "waveform_codec_lossless_decode_channel",
)
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass
class WaveformCompressionResult:
    """Result of waveform compression."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    n_channels: int
    n_samples: int
    n_spikes_detected: int
    n_templates: int
    spike_bytes: int
    snippet_bytes: int
    background_bytes: int
    lossless_spikes: bool
    lossless_waveform: bool = False


@dataclass
class WaveformDecodeResult:
    """Result of ``WaveformCodec.decompress``.

    Attributes
    ----------
    mode : str
        Mode the stream was written in.
    spike_raster : ndarray of shape (T, N), int8, or None
        Detected spikes, exact; ``None`` for a ``lossless`` stream, which
        detects none.
    waveform : ndarray of shape (T, N) or None
        ``lossless``: the input samples exactly, int16. ``full``: background
        plus snippet reconstruction, float32 (lossy). ``waveform``: snippet
        reconstruction on zeros, float32 (lossy). ``spike``: ``None``.
    lossless_waveform : bool
        True only when ``waveform`` equals the input sample for sample.
    """

    mode: str
    spike_raster: np.ndarray[Any, Any] | None
    waveform: np.ndarray[Any, Any] | None
    lossless_waveform: bool


class WaveformCodec:
    """End-to-end neural waveform codec.

    Pipeline: detect → separate → compress each component optimally.

    Parameters
    ----------
    threshold_sigma : float
        Spike detection threshold in units of per-channel noise sigma.
        Must be finite and positive. Typical: 4.0-5.0 (4 sigma catches
        ~99.99% of noise).
    snippet_samples : int
        Waveform samples to extract around each spike (before + after peak).
        Must fit the one-byte wire header: 1-255 samples.
    max_templates : int
        Maximum number of spike waveform templates to maintain.
        Must fit the two-byte wire header: 1-65535 templates.
    template_threshold : float
        Correlation threshold for template matching, inclusive range 0-1.
    quantize_bits : int
        Background signal quantization, inclusive range 1-8. Fewer bits
        increase compression.
    mode : str
        Compression mode controlling what is preserved:
        - ``"full"``: spike timing + waveform templates + background LFP (~137x)
        - ``"waveform"``: spike timing + waveform templates, no background (~1700x)
        - ``"spike"``: spike timing only, Neuralink-equivalent (~4500x)
        - ``"lossless"``: every input sample exactly (integer input only)
    backend : str
        ``"auto"`` (default: Rust kernels when installed, else NumPy),
        ``"rust"`` (refuses when the engine lacks the kernels) or
        ``"python"``. Streams are byte-identical across backends.

        The ratios above were measured on synthetic data (2026-04-07). On the
        public Neuralink compression-challenge recordings the lossy ratios are
        different and depend on spike detection; see the benchmark records.
    """

    HEADER_MAGIC = b"WFC2"
    LEGACY_HEADER_MAGIC = b"WFCX"

    def __init__(
        self,
        threshold_sigma: float = 4.5,
        snippet_samples: int = 48,
        max_templates: int = 16,
        template_threshold: float = 0.9,
        quantize_bits: int = 6,
        mode: str = "full",
        backend: str = "auto",
    ):
        if mode not in WAVEFORM_CODEC_VALID_MODES:
            raise ValueError(
                f"mode must be 'full', 'waveform', 'spike' or 'lossless', got {mode!r}"
            )
        self.threshold_sigma = self._require_positive_float("threshold_sigma", threshold_sigma)
        self.snippet_samples = self._require_int_range(
            "snippet_samples",
            snippet_samples,
            minimum=WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES,
            maximum=WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES,
        )
        self.max_templates = self._require_int_range(
            "max_templates",
            max_templates,
            minimum=WAVEFORM_CODEC_MIN_TEMPLATES,
            maximum=WAVEFORM_CODEC_MAX_TEMPLATES,
        )
        self.template_threshold = self._require_float_range(
            "template_threshold", template_threshold, minimum=0.0, maximum=1.0
        )
        self.quantize_bits = self._require_int_range(
            "quantize_bits",
            quantize_bits,
            minimum=WAVEFORM_CODEC_MIN_QUANTIZE_BITS,
            maximum=WAVEFORM_CODEC_MAX_QUANTIZE_BITS,
        )
        self.mode = mode
        self.backend = backend
        self._native = _native_kernels(backend)
        self.spike_codec = SpikeCodec(entropy="auto")

    @staticmethod
    def _require_positive_float(name: str, value: float) -> float:
        """Return a finite positive float or raise a field-specific error."""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite positive float, got {value!r}")
        numeric = float(value)
        if not isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be a finite positive float, got {value!r}")
        return numeric

    @staticmethod
    def _require_float_range(
        name: str,
        value: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        """Return a finite float inside an inclusive range."""
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be a finite float in [{minimum}, {maximum}], got {value!r}"
            )
        numeric = float(value)
        if not isfinite(numeric) or numeric < minimum or numeric > maximum:
            raise ValueError(
                f"{name} must be a finite float in [{minimum}, {maximum}], got {value!r}"
            )
        return numeric

    @staticmethod
    def _require_int_range(
        name: str,
        value: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        """Return an integer inside an inclusive range."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}")
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}")
        return value

    def compress(self, waveform: np.ndarray[Any, Any]) -> tuple[bytes, WaveformCompressionResult]:
        """Compress raw electrode waveform.

        Parameters
        ----------
        waveform : ndarray of shape (T, N), int16 or float
            Raw ADC samples. T = timesteps, N = channels.

        Returns
        -------
        (compressed_bytes, WaveformCompressionResult)

        Raises
        ------
        ValueError
            If the array is not a finite non-empty matrix, or, in ``lossless``
            mode, if a sample is not an integer in the int16 range.
        """
        if self.mode == "lossless":
            return self._compress_lossless(waveform)
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = self._validate_waveform(waveform)
        T, N = waveform.shape
        original_bytes = T * N * 2  # 16-bit raw

        # Step 1: Per-channel noise estimation (MAD estimator) → negative thresholds
        thresholds = self._noise_thresholds(waveform)

        # Step 2: Threshold-crossing spike detection
        spike_raster, spike_times_per_ch = self._detect_spikes(waveform, thresholds)

        # Step 3: Extract spike snippets
        snippets, snippet_indices = self._extract_snippets(waveform, spike_times_per_ch, N)

        # Step 4: Template matching on snippets
        templates, template_ids, scales, residuals = self._template_match(snippets)

        # Step 5: Compress spike timing (binary raster → ISI)
        spike_data, _ = self.spike_codec.compress(spike_raster)

        # Step 6: Compress templates + template IDs + residuals
        # (skipped in "spike" mode — only timing is kept)
        snippet_data = (
            self._compress_snippets(templates, template_ids, scales, residuals)
            if self.mode != "spike"
            else b""
        )

        # Step 7: Compress background (waveform minus spikes)
        # (only in "full" mode — BCI decoders rarely need LFP)
        if self.mode == "full":
            background = self._extract_background(waveform, spike_times_per_ch)
            bg_data = self._compress_background(background)
        else:
            bg_data = b""

        # Pack everything
        if len(snippet_indices) > WAVEFORM_CODEC_MAX_HEADER_COUNT:
            raise ValueError("spike snippet count exceeds WaveformCodec header capacity")
        header = self._header(T, N, len(templates), len(snippet_indices), len(spike_data))
        # Length-prefixed sections
        parts = [
            header,
            struct.pack("!I", len(spike_data)),
            spike_data,
            struct.pack("!I", len(snippet_data)),
            snippet_data,
            struct.pack("!I", len(bg_data)),
            bg_data,
        ]
        encoded = _seal(b"".join(parts))

        n_spikes = int(spike_raster.sum())

        return encoded, WaveformCompressionResult(
            original_bytes=original_bytes,
            compressed_bytes=len(encoded),
            compression_ratio=original_bytes / max(len(encoded), 1),
            n_channels=N,
            n_samples=T,
            n_spikes_detected=n_spikes,
            n_templates=len(templates),
            spike_bytes=len(spike_data),
            snippet_bytes=len(snippet_data),
            background_bytes=len(bg_data),
            lossless_spikes=True,
        )

    @staticmethod
    def _validate_waveform(waveform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return a finite non-empty ``(time, channel)`` waveform matrix."""
        if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise ValueError(
                "waveform must be a finite two-dimensional array with at least "
                "one sample and one channel"
            )
        if not bool(np.isfinite(waveform).all()):
            raise ValueError("waveform samples must be finite")
        return waveform

    def _noise_thresholds(self, waveform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return float32 thresholds ``-threshold_sigma * max(median(|x|) / 0.6745, 1e-6)`` per channel."""
        if self._native is not None:
            return np.asarray(
                self._native.waveform_codec_noise_thresholds(
                    np.ascontiguousarray(waveform, dtype=np.float32), -self.threshold_sigma
                ),
                dtype=np.float32,
            )
        noise_sigma = np.median(np.abs(waveform), axis=0) / 0.6745
        noise_sigma = np.maximum(noise_sigma, 1e-6)
        return np.asarray(-self.threshold_sigma * noise_sigma, dtype=np.float32)

    def _detect_spikes(
        self, waveform: np.ndarray[Any, Any], thresholds: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], list[list[int]]]:
        """Threshold-crossing spike detection with refractory period."""
        T, N = waveform.shape
        raster = np.zeros((T, N), dtype=np.int8)
        refractory = self.snippet_samples // 2
        if self._native is not None:
            native_times: list[list[int]] = self._native.waveform_codec_detect_spikes(
                np.ascontiguousarray(waveform, dtype=np.float32),
                np.ascontiguousarray(thresholds, dtype=np.float32),
                refractory,
            )
            for ch, channel_times in enumerate(native_times):
                raster[channel_times, ch] = 1
            return raster, native_times
        times_per_ch: list[list[int]] = [[] for _ in range(N)]

        for ch in range(N):
            last_spike = -refractory - 1
            for t in range(1, T):
                if (
                    waveform[t, ch] < thresholds[ch]
                    and waveform[t, ch] < waveform[t - 1, ch]
                    and (t - last_spike) > refractory
                ):
                    raster[t, ch] = 1
                    times_per_ch[ch].append(t)
                    last_spike = t

        return raster, times_per_ch

    def _extract_snippets(
        self, waveform: np.ndarray[Any, Any], times_per_ch: list[list[int]], N: int
    ) -> tuple[list[np.ndarray[Any, Any]], list[tuple[int, int]]]:
        """Extract waveform clips around detected spikes."""
        T = waveform.shape[0]
        half = self.snippet_samples // 2
        snippets = []
        indices = []

        for ch in range(N):
            for t in times_per_ch[ch]:
                start = max(0, t - half)
                end = min(T, t + half)
                clip = waveform[start:end, ch]
                if len(clip) < self.snippet_samples:
                    clip = np.pad(clip, (0, self.snippet_samples - len(clip)))
                else:
                    clip = clip[: self.snippet_samples]
                snippets.append(clip.astype(np.float32))
                indices.append((ch, t))

        return snippets, indices

    def _template_match(
        self, snippets: list[np.ndarray[Any, Any]]
    ) -> tuple[list[np.ndarray[Any, Any]], list[int], list[float], list[np.ndarray[Any, Any]]]:
        """Build template library and match snippets.

        Every snippet is ``scale * templates[template_id] + residual``; the
        scale is 1 except for a snippet matched above ``template_threshold``.
        """
        if not snippets:
            return [], [], [], []
        if self._native is not None:
            sources, native_ids, native_scales, native_residuals = (
                self._native.waveform_codec_template_match(
                    np.ascontiguousarray(np.stack(snippets), dtype=np.float32),
                    self.max_templates,
                    self.template_threshold,
                )
            )
            return (
                [snippets[i].copy() for i in sources],
                list(native_ids),
                list(native_scales),
                list(native_residuals),
            )

        templates = [snippets[0].copy()]
        template_ids = [0]
        scales = [1.0]
        residuals = [np.zeros_like(snippets[0])]

        for i in range(1, len(snippets)):
            s = snippets[i]
            best_corr = -1.0
            best_idx = -1

            for j, tmpl in enumerate(templates):
                norm_s = math.sqrt(_exact_dot(s, s))
                norm_t = math.sqrt(_exact_dot(tmpl, tmpl))
                if norm_s > 1e-6 and norm_t > 1e-6:
                    corr = _exact_dot(s, tmpl) / (norm_s * norm_t)
                    if corr > best_corr:
                        best_corr = corr
                        best_idx = j

            if best_corr >= self.template_threshold:
                template_ids.append(best_idx)
                scale = _exact_dot(s, templates[best_idx]) / max(
                    _exact_dot(templates[best_idx], templates[best_idx]), 1e-10
                )
                scales.append(float(scale))
                residuals.append(s - np.float32(scale) * templates[best_idx])
            elif len(templates) < self.max_templates:
                templates.append(s.copy())
                template_ids.append(len(templates) - 1)
                scales.append(1.0)
                residuals.append(np.zeros_like(s))
            else:
                # A snippet with no measurable correlation (zero norm) falls back
                # to template 0; ``best_idx`` would otherwise be -1.
                fallback = max(best_idx, 0)
                template_ids.append(fallback)
                scales.append(1.0)
                residuals.append(s - templates[fallback])

        return templates, template_ids, scales, residuals

    def _compress_snippets(
        self,
        templates: list[np.ndarray[Any, Any]],
        template_ids: list[int],
        scales: list[float],
        residuals: list[np.ndarray[Any, Any]],
    ) -> bytes:
        """Compress templates + IDs + per-snippet scales + quantised residuals."""
        try:
            import zstandard as zstd

            def _zstd_compress(data: bytes) -> bytes:
                return bytes(zstd.ZstdCompressor(level=19).compress(data))
        except ImportError:
            import zlib

            def _zstd_compress(data: bytes) -> bytes:
                return zlib.compress(data, 9)

        parts = []

        # Templates: quantise float32 → int8 (4x savings per template)
        parts.append(struct.pack("!H", len(templates)))
        if templates:
            tmpl_arr = np.array(templates, dtype=np.float32)
            tmpl_max = max(np.abs(tmpl_arr).max(), 1e-6)
            tmpl_q = np.clip(tmpl_arr / tmpl_max * 127, -127, 127).astype(np.int8)
            parts.append(struct.pack("!f", float(tmpl_max)))
            parts.append(tmpl_q.tobytes())
        else:
            parts.append(struct.pack("!f", 0.0))

        # Template IDs: varint per spike
        parts.append(struct.pack("!I", len(template_ids)))
        for tid in template_ids:
            parts.append(SpikeCodec._encode_varint(tid))

        # Per-snippet template scales (big-endian float32)
        parts.append(np.asarray(scales, dtype=">f4").tobytes())

        # Residuals: quantise to 4-bit, nibble-pack, then zstd
        if residuals:
            all_res = np.array(residuals)
            res_max = max(np.abs(all_res).max(), 1e-6)
            quantized = np.clip(np.round(all_res / res_max * 7), -7, 7).astype(np.int8)
            # Nibble-pack: two int4 values per byte
            flat = (quantized.flatten() + 8).astype(np.uint8)  # shift to 0-15
            if len(flat) % 2:
                flat = np.append(flat, np.uint8(8))  # pad with zero
            packed = (flat[0::2] << 4) | flat[1::2]
            compressed = _zstd_compress(packed.tobytes())
            parts.append(struct.pack("!fI", float(res_max), len(flat)))
            parts.append(compressed)
        else:
            parts.append(struct.pack("!fI", 0.0, 0))

        return b"".join(parts)

    def _extract_background(
        self, waveform: np.ndarray[Any, Any], times_per_ch: list[list[int]]
    ) -> np.ndarray[Any, Any]:
        """Zero spike windows and average the background over blocks of 16 rows.

        Each block average is the float32 sum of its rows in order divided by
        16 (the same operations as the Rust kernel); fewer than 16 rows are
        returned whole.
        """
        T, N = waveform.shape
        half = self.snippet_samples // 2
        if self._native is not None:
            return np.asarray(
                self._native.waveform_codec_extract_background(
                    np.ascontiguousarray(waveform, dtype=np.float32), times_per_ch, half
                ),
                dtype=np.float32,
            )
        bg = waveform.copy()
        for ch in range(N):
            for t in times_per_ch[ch]:
                start = max(0, t - half)
                end = min(T, t + half)
                bg[start:end, ch] = 0  # zero out spike regions

        # Downsample by 16 (the background does not need 20 kHz)
        ds = WAVEFORM_CODEC_BACKGROUND_DOWNSAMPLE
        if ds > T:
            return bg
        blocks = bg[: T - T % ds].reshape(-1, ds, N)
        acc = blocks[:, 0, :].copy()
        for k in range(1, ds):
            acc = acc + blocks[:, k, :]
        return np.asarray(acc / np.float32(ds), dtype=np.float32)

    def _compress_background(self, background: np.ndarray[Any, Any]) -> bytes:
        """Delta-encode + quantize background signal."""
        if background.size == 0:
            return b""

        # Spatial decorrelation: subtract adjacent channel (exploits LFP
        # volume conduction correlation on Neuropixels/Utah arrays)
        if background.shape[1] > 1:
            spatial_ref = np.empty_like(background)
            spatial_ref[:, 0] = background[:, 0]
            spatial_ref[:, 1:] = background[:, 1:] - background[:, :-1]
            background = spatial_ref

        # Wavelet denoising (optional — requires PyWavelets)
        try:
            import pywt

            original_len = background.shape[0]
            coeffs = pywt.wavedec(background, "db4", axis=0)
            # Calibrated: threshold=3.0 gives SNR ≥24 dB, energy retained ≥99.7%
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], 3.0, mode="hard")
            background = pywt.waverec(coeffs, "db4", axis=0)[:original_len]
        except ImportError:
            pass  # Skip wavelet denoising if PyWavelets not installed

        # Closed-loop temporal delta quantisation: each step is quantised
        # against the decoder's reconstruction, so the error stays within one
        # quantisation step instead of accumulating. The first row is sent as
        # float32, so the absolute level is kept.
        background = np.asarray(background, dtype=np.float32)
        first = background[0].astype(np.float32)
        delta = np.diff(background, axis=0)
        # Rounded to float32 first: the decoder derives the step from the stored float32.
        dmax = float(np.float32(max(np.abs(delta).max(), 1e-6))) if delta.size else 1e-6
        half = (1 << self.quantize_bits) // 2
        step = _background_step(dmax, self.quantize_bits)
        if self._native is not None:
            quantized = self._native.waveform_codec_quantize_background(
                np.ascontiguousarray(background, dtype=np.float32), float(step), half
            )
        else:
            quantized = np.zeros_like(background, dtype=np.int8)
            recon = first.copy()
            for t in range(1, background.shape[0]):
                q = np.clip(np.round((background[t] - recon) / step), -half, half - 1)
                quantized[t] = q.astype(np.int8)
                recon = recon + q.astype(np.float32) * step

        raw_bytes = quantized.tobytes()
        try:
            import zstandard as zstd

            compressed = bytes(zstd.ZstdCompressor(level=19).compress(raw_bytes))
        except ImportError:
            import zlib

            compressed = zlib.compress(raw_bytes, 9)
        header = struct.pack("!IIf", background.shape[0], background.shape[1], float(dmax))
        return header + bytes(first.astype(">f4").tobytes()) + compressed

    def _header(self, T: int, N: int, n_templates: int, n_snippets: int, spike_bytes: int) -> bytes:
        """Pack the stream header: magic, dimensions, counts and mode."""
        return self.HEADER_MAGIC + struct.pack(
            _HEADER_FORMAT,
            T,
            N,
            n_templates,
            n_snippets,
            self.snippet_samples,
            self.quantize_bits,
            spike_bytes.bit_length(),
            WAVEFORM_CODEC_MODE_BYTES[self.mode],
        )

    def _compress_lossless(
        self, waveform: np.ndarray[Any, Any]
    ) -> tuple[bytes, WaveformCompressionResult]:
        """Encode every sample exactly: lattice, fixed predictor, adaptive Rice codes."""
        raw = np.asarray(waveform)
        checked = self._validate_waveform(np.asarray(raw, dtype=np.float64))
        if not bool(np.all(checked == np.round(checked))):
            raise ValueError("lossless mode needs integer samples")
        if checked.min() < -32768 or checked.max() > 32767:
            raise ValueError("lossless mode needs samples in the int16 range")
        samples = checked.astype(np.int64)
        T, N = samples.shape
        if self._native is not None:
            channels = [
                bytes(
                    self._native.waveform_codec_lossless_encode_channel(
                        np.ascontiguousarray(samples[:, ch], dtype=np.int64)
                    )
                )
                for ch in range(N)
            ]
        else:
            channels = [_encode_lossless_channel(samples[:, ch]) for ch in range(N)]
        body = b"".join(struct.pack("!I", len(c)) + c for c in channels)
        encoded = _seal(self._header(T, N, 0, 0, 0) + struct.pack("!I", len(body)) + body)
        original_bytes = T * N * 2
        return encoded, WaveformCompressionResult(
            original_bytes=original_bytes,
            compressed_bytes=len(encoded),
            compression_ratio=original_bytes / max(len(encoded), 1),
            n_channels=N,
            n_samples=T,
            n_spikes_detected=0,
            n_templates=0,
            spike_bytes=0,
            snippet_bytes=0,
            background_bytes=0,
            lossless_spikes=False,
            lossless_waveform=True,
        )

    @classmethod
    def decompress(
        cls,
        data: bytes,
        *,
        max_samples: int = WAVEFORM_CODEC_MAX_DECODE_SAMPLES,
        backend: str = "auto",
    ) -> WaveformDecodeResult:
        """Decode a stream written by ``compress``.

        Parameters
        ----------
        data : bytes
            A ``WFC2`` stream, or a legacy ``WFCX`` stream (spike raster only).
        max_samples : int
            Largest ``T * N`` the stream may declare; checked before any
            allocation. Default ``2**28``.
        backend : str
            ``"auto"``, ``"rust"`` or ``"python"`` (see the class); results are
            identical.

        Returns
        -------
        WaveformDecodeResult
            Mode, spike raster and the reconstructed waveform (see the class).

        Raises
        ------
        ValueError
            If the stream is not a valid, intact WaveformCodec stream within
            ``max_samples``: wrong magic, checksum, mode, dimensions, section
            lengths or section contents.
        """
        native = _native_kernels(backend)
        try:
            return cls._decompress(bytes(data), max_samples, native)
        except (IndexError, OverflowError, struct.error) as exc:
            raise ValueError(f"malformed WaveformCodec stream: {exc}") from exc

    @classmethod
    def _decompress(
        cls, data: bytes, max_samples: int, native: ModuleType | None
    ) -> WaveformDecodeResult:
        """Decode an intact stream; ``decompress`` maps parse errors to ValueError."""
        header_size = 4 + struct.calcsize(_HEADER_FORMAT)
        if len(data) < header_size:
            raise ValueError("stream shorter than the WaveformCodec header")
        magic = data[:4]
        if magic not in (cls.HEADER_MAGIC, cls.LEGACY_HEADER_MAGIC):
            raise ValueError(f"not a WaveformCodec stream (magic {magic!r})")
        if magic == cls.HEADER_MAGIC:
            if len(data) < header_size + 4:
                raise ValueError("stream shorter than the WaveformCodec header and checksum")
            body, (crc,) = data[:-4], struct.unpack("!I", data[-4:])
            if zlib.crc32(body) != crc:
                raise ValueError("WaveformCodec checksum mismatch: the stream is corrupted")
            data = body
        T, N, n_templates, n_snippets, snippet_samples, quantize_bits, _, mode_byte = struct.unpack(
            _HEADER_FORMAT, data[4:header_size]
        )
        mode = WAVEFORM_CODEC_MODES_BY_BYTE.get(mode_byte)
        if mode is None:
            raise ValueError(f"unknown WaveformCodec mode byte {mode_byte}")
        if T < 1 or N < 1 or max_samples < T * N:
            raise ValueError(
                f"stream declares {T} x {N} samples; at least one each and at most "
                f"max_samples={max_samples} in total"
            )
        reader = _SectionReader(data, header_size)
        if mode == "lossless":
            if magic != cls.HEADER_MAGIC:
                raise ValueError("lossless streams exist only in the WFC2 format")
            body_reader = _SectionReader(reader.section(), 0)
            if native is not None:
                columns = [
                    native.waveform_codec_lossless_decode_channel(body_reader.section(), T)
                    for _ in range(N)
                ]
            else:
                columns = [_decode_lossless_channel(body_reader.section(), T) for _ in range(N)]
            body_reader.finish()
            reader.finish()
            waveform = np.stack(columns, axis=1).astype(np.int16)
            return WaveformDecodeResult(mode, None, waveform, True)

        spike_section = reader.section()
        snippet_section = reader.section()
        background_section = reader.section()
        reader.finish()
        raster = SpikeCodec(entropy="auto").decompress(spike_section, T, N)
        if int(raster.sum()) != n_snippets:
            raise ValueError("snippet count does not match the decoded spike raster")
        if mode == "spike":
            return WaveformDecodeResult(mode, raster, None, False)
        if magic != cls.HEADER_MAGIC:
            raise ValueError(
                "legacy WFCX streams lack per-snippet scales and the background's first row; "
                "only their spike raster can be decoded (use mode 'spike' semantics)"
            )
        if mode == "full":
            waveform = _decode_background(background_section, T, N, quantize_bits, native)
        else:
            waveform = np.zeros((T, N), dtype=np.float32)
        snippets = _decode_snippets(snippet_section, n_templates, n_snippets, snippet_samples)
        half = snippet_samples // 2
        index = 0
        for ch in range(N):
            for t in np.flatnonzero(raster[:, ch]):
                start = max(0, int(t) - half)
                end = min(T, int(t) + half)
                width = min(end - start, snippet_samples)
                waveform[start : start + width, ch] = snippets[index][:width]
                index += 1
        return WaveformDecodeResult(mode, raster, waveform, False)


def _native_kernels(backend: str) -> ModuleType | None:
    """Return the engine module providing every WaveformCodec kernel, or None for NumPy.

    Raises
    ------
    ValueError
        For an unknown backend, or ``"rust"`` when the kernels are not installed.
    """
    if backend not in WAVEFORM_CODEC_BACKENDS:
        raise ValueError(f"backend must be one of {WAVEFORM_CODEC_BACKENDS}, got {backend!r}")
    if backend == "python":
        return None
    try:
        engine: ModuleType | None = importlib.import_module(
            "sc_neurocore_engine.sc_neurocore_engine"
        )
    except ImportError:
        engine = None
    if engine is not None and all(hasattr(engine, name) for name in _NATIVE_KERNELS):
        return engine
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the WaveformCodec kernels")
    return None


def _exact_dot(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Correctly rounded dot product of exact float64 products (matches the Rust kernel)."""
    products = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    return math.fsum(products.tolist())


def _seal(stream: bytes) -> bytes:
    """Append the CRC-32 of ``stream`` (big-endian) as the WFC2 trailer."""
    return stream + struct.pack("!I", zlib.crc32(stream))


class _SectionReader:
    """Read length-prefixed (``!I``) sections from a byte string."""

    def __init__(self, data: bytes, position: int) -> None:
        self.data = data
        self.position = position

    def section(self) -> bytes:
        """Return the next section and advance past it."""
        if self.position + 4 > len(self.data):
            raise ValueError("truncated WaveformCodec stream")
        (length,) = struct.unpack("!I", self.data[self.position : self.position + 4])
        start = self.position + 4
        if start + length > len(self.data):
            raise ValueError("truncated WaveformCodec stream")
        self.position = start + length
        return self.data[start : self.position]

    def finish(self) -> None:
        """Refuse trailing bytes after the last section."""
        if self.position != len(self.data):
            raise ValueError("trailing bytes after the WaveformCodec stream")


def _inflate(blob: bytes) -> bytes:
    """Decompress a zstd frame or a zlib stream, recognised by its magic."""
    try:
        if blob[:4] == _ZSTD_MAGIC:
            import zstandard as zstd

            return bytes(zstd.ZstdDecompressor().decompress(blob))
        return zlib.decompress(blob)
    except ImportError as exc:
        raise ValueError("a zstd-compressed section needs the zstandard package") from exc
    except Exception as exc:  # zlib.error and zstandard.ZstdError share no base class
        raise ValueError(f"compressed WaveformCodec section is corrupted: {exc}") from exc


def _decode_snippets(
    section: bytes, n_templates: int, n_snippets: int, snippet_samples: int
) -> list[np.ndarray[Any, Any]]:
    """Rebuild every snippet as ``scale * template + residual`` (lossy)."""
    pos = 0
    (count,) = struct.unpack("!H", section[pos : pos + 2])
    pos += 2
    (tmpl_max,) = struct.unpack("!f", section[pos : pos + 4])
    pos += 4
    if count != n_templates:
        raise ValueError("template count disagrees with the header")
    size = count * snippet_samples
    templates = np.frombuffer(section[pos : pos + size], dtype=np.int8).astype(np.float32).reshape(
        count, snippet_samples
    ) * np.float32(tmpl_max / 127.0)
    pos += size
    (n_ids,) = struct.unpack("!I", section[pos : pos + 4])
    pos += 4
    if n_ids != n_snippets:
        raise ValueError("snippet count disagrees with the header")
    ids = []
    for _ in range(n_ids):
        value, pos = SpikeCodec._decode_varint(section, pos)
        ids.append(value)
    scales = np.frombuffer(section[pos : pos + 4 * n_ids], dtype=">f4").astype(np.float32)
    pos += 4 * n_ids
    res_max, n_flat = struct.unpack("!fI", section[pos : pos + 8])
    pos += 8
    if n_flat:
        packed = np.frombuffer(_inflate(section[pos:]), dtype=np.uint8)
        nibbles = np.empty(packed.size * 2, dtype=np.int16)
        nibbles[0::2] = packed >> 4
        nibbles[1::2] = packed & 0x0F
        residuals = (nibbles[:n_flat] - 8).astype(np.float32) * np.float32(res_max / 7.0)
        residuals = residuals[: n_ids * snippet_samples].reshape(n_ids, snippet_samples)
    else:
        residuals = np.zeros((n_ids, snippet_samples), dtype=np.float32)
    return [scales[i] * templates[ids[i]] + residuals[i] for i in range(n_ids)]


def _decode_background(
    section: bytes, T: int, N: int, quantize_bits: int, native: ModuleType | None
) -> np.ndarray[Any, Any]:
    """Rebuild the full-rate background from its closed-loop quantised deltas (lossy)."""
    waveform = np.zeros((T, N), dtype=np.float32)
    if len(section) < 12:
        raise ValueError("full-mode stream lacks its background section")
    rows, cols, dmax = struct.unpack("!IIf", section[:12])
    first = np.frombuffer(section[12 : 12 + 4 * cols], dtype=">f4").astype(np.float32)
    quantized = np.frombuffer(_inflate(section[12 + 4 * cols :]), dtype=np.int8).reshape(rows, cols)
    step = _background_step(dmax, quantize_bits)
    recon: np.ndarray[Any, Any]
    if native is not None:
        recon = np.asarray(
            native.waveform_codec_reconstruct_background(
                np.ascontiguousarray(quantized), np.ascontiguousarray(first), float(step)
            ),
            dtype=np.float32,
        )
    else:
        recon = np.empty((rows, cols), dtype=np.float32)
        recon[0] = first
        for t in range(1, rows):
            recon[t] = recon[t - 1] + quantized[t].astype(np.float32) * step
    if cols > 1:
        recon = np.cumsum(recon, axis=1, dtype=np.float32)
    ds = WAVEFORM_CODEC_BACKGROUND_DOWNSAMPLE
    if rows == T:
        return recon
    full = np.repeat(recon, ds, axis=0)
    waveform[: full.shape[0]] = full
    waveform[full.shape[0] :] = recon[-1]
    return waveform


def _background_step(dmax: float, quantize_bits: int) -> np.float32:
    """Quantisation step that maps the largest delta onto the largest positive code.

    The codes run from ``-half`` to ``half - 1``; scaling by ``half - 1`` (at
    least 1) keeps the largest delta representable, so the closed loop does
    not clip on steep segments.
    """
    return np.float32(dmax / max((1 << quantize_bits) // 2 - 1, 1))


def _zigzag(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Map signed integers onto non-negative ones (0, -1, 1, -2 to 0, 1, 2, 3)."""
    return np.where(values >= 0, 2 * values, -2 * values - 1).astype(np.int64)


def _unzigzag(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Invert ``_zigzag``."""
    return np.where(values % 2 == 0, values // 2, -(values + 1) // 2).astype(np.int64)


def _encode_lossless_channel(x: np.ndarray[Any, Any]) -> bytes:
    """Encode one channel of integer samples exactly.

    The samples are first mapped onto small integer levels, by whichever of
    two mappings gives the shorter stream: an affine lattice
    ``offset + step * level`` (``step`` the gcd of the offsets), or the index
    into the channel's sorted table of used values (useful when a 10-bit ADC
    is scaled to 16 bits by a non-integer factor, so no lattice exists).
    """
    offset = int(x.min())
    step = int(np.gcd.reduce(x - offset)) or 1
    affine = struct.pack("!BiI", 0, offset, step) + _encode_levels((x - offset) // step)
    table = np.unique(x)
    indexed = (
        struct.pack("!BI", 1, len(table))
        + b"".join(SpikeCodec._encode_varint(int(v)) for v in _zigzag(np.diff(table, prepend=0)))
        + _encode_levels(np.searchsorted(table, x).astype(np.int64))
    )
    return indexed if len(indexed) < len(affine) else affine


def _encode_levels(levels: np.ndarray[Any, Any]) -> bytes:
    """Predict integer levels with the best fixed polynomial and Rice-code the residuals."""
    order = min(
        range(min(WAVEFORM_CODEC_LOSSLESS_MAX_ORDER, len(levels) - 1) + 1),
        key=lambda k: int(np.abs(np.diff(levels, n=k)).sum()),
    )
    firsts = [int(np.diff(levels, n=j)[0]) for j in range(order)]
    residuals = _zigzag(np.diff(levels, n=order))
    head = struct.pack("!B", order) + b"".join(
        SpikeCodec._encode_varint(int(v)) for v in _zigzag(np.asarray(firsts, dtype=np.int64))
    )
    return head + _rice_encode(residuals)


def _decode_lossless_channel(section: bytes, T: int) -> np.ndarray[Any, Any]:
    """Decode one channel written by ``_encode_lossless_channel``."""
    kind = section[0]
    if kind == 1:
        (n_table,) = struct.unpack("!I", section[1:5])
        pos = 5
        if not 1 <= n_table <= len(section) - pos:
            raise ValueError("lossless value table length is impossible for its section")
        steps = []
        for _ in range(n_table):
            value, pos = SpikeCodec._decode_varint(section, pos)
            steps.append(value)
        table = np.cumsum(_unzigzag(np.asarray(steps, dtype=np.int64)))
    elif kind == 0:
        offset, step = struct.unpack("!iI", section[1:9])
        pos = 9
    else:
        raise ValueError("invalid lossless level mapping")
    order = section[pos]
    pos += 1
    if order > WAVEFORM_CODEC_LOSSLESS_MAX_ORDER or order >= max(T, 1):
        raise ValueError("invalid lossless predictor order")
    firsts = []
    for _ in range(order):
        value, pos = SpikeCodec._decode_varint(section, pos)
        firsts.append(value)
    current = _unzigzag(_rice_decode(section[pos:], T - order))
    for first in reversed(_unzigzag(np.asarray(firsts, dtype=np.int64)).tolist()):
        current = np.concatenate(([first], first + np.cumsum(current)))
    if kind == 1:
        if current.min() < 0 or current.max() >= len(table):
            raise ValueError("lossless level outside the value table")
        samples = np.asarray(table[current], dtype=np.int64)
    else:
        samples = np.asarray(current * step + offset, dtype=np.int64)
    if samples.min() < -32768 or samples.max() > 32767:
        raise ValueError("lossless samples outside the int16 range")
    return samples


def _rice_encode(u: np.ndarray[Any, Any]) -> bytes:
    """Block-adaptive Rice coding of non-negative integers in three bit streams."""
    n = len(u)
    block = WAVEFORM_CODEC_LOSSLESS_BLOCK
    escape = WAVEFORM_CODEC_LOSSLESS_ESCAPE
    n_blocks = -(-n // block)
    padded = np.zeros(n_blocks * block, dtype=np.int64)
    padded[:n] = u
    valid = (np.arange(n_blocks * block) < n).reshape(n_blocks, block)
    candidates = np.arange(WAVEFORM_CODEC_LOSSLESS_MAX_RICE_K + 1)
    quotients = padded.reshape(n_blocks, block)[:, :, None] >> candidates
    plain = quotients < escape
    cost = np.where(plain, quotients + 1 + candidates, escape + 1 + 32) * valid[:, :, None]
    ks = cost.sum(axis=1).argmin(axis=1).astype(np.uint8)
    k = np.repeat(ks.astype(np.int64), block)[:n]
    q = u >> k
    is_plain = q < escape
    unary_len = np.where(is_plain, q, escape) + 1
    unary = np.ones(int(unary_len.sum()), dtype=np.uint8)
    unary[np.cumsum(unary_len) - 1] = 0
    width = np.where(is_plain, k, 0)
    starts = np.cumsum(width) - width
    owner = np.repeat(np.arange(n), width)
    bit_index = np.arange(int(width.sum())) - np.repeat(starts, width)
    remainder_bits = ((u[owner] >> (width[owner] - 1 - bit_index)) & 1).astype(np.uint8)
    escapes = u[~is_plain].astype(">u4")
    return (
        struct.pack("!III", len(unary), len(remainder_bits), len(escapes))
        + ks.tobytes()
        + np.packbits(unary).tobytes()
        + np.packbits(remainder_bits).tobytes()
        + escapes.tobytes()
    )


def _rice_decode(data: bytes, n: int) -> np.ndarray[Any, Any]:
    """Invert ``_rice_encode`` for ``n`` values."""
    block = WAVEFORM_CODEC_LOSSLESS_BLOCK
    escape = WAVEFORM_CODEC_LOSSLESS_ESCAPE
    n_unary, n_rem, n_esc = struct.unpack("!III", data[:12])
    n_blocks = -(-n // block)
    pos = 12
    ks = np.frombuffer(data[pos : pos + n_blocks], dtype=np.uint8).astype(np.int64)
    pos += n_blocks
    unary_bytes = -(-n_unary // 8)
    unary = np.unpackbits(np.frombuffer(data[pos : pos + unary_bytes], dtype=np.uint8))[:n_unary]
    pos += unary_bytes
    rem_bytes = -(-n_rem // 8)
    rem_bits = np.unpackbits(np.frombuffer(data[pos : pos + rem_bytes], dtype=np.uint8))[:n_rem]
    pos += rem_bytes
    escapes = np.frombuffer(data[pos : pos + 4 * n_esc], dtype=">u4").astype(np.int64)
    if pos + 4 * n_esc != len(data):
        raise ValueError("lossless Rice section has the wrong length")
    if ks.size and int(ks.max()) > WAVEFORM_CODEC_LOSSLESS_MAX_RICE_K:
        raise ValueError("lossless Rice parameter out of range")
    zeros = np.flatnonzero(unary == 0)
    if len(zeros) != n:
        raise ValueError("lossless Rice section has the wrong number of values")
    q = np.diff(np.concatenate(([-1], zeros))) - 1
    k = np.repeat(ks, block)[:n]
    is_plain = q < escape
    width = np.where(is_plain, k, 0)
    if int(width.sum()) != n_rem or int((~is_plain).sum()) != n_esc:
        raise ValueError("lossless Rice section is inconsistent")
    owner = np.repeat(np.arange(n), width)
    bit_index = np.arange(n_rem) - np.repeat(np.cumsum(width) - width, width)
    weights = rem_bits.astype(np.int64) << (width[owner] - 1 - bit_index)
    remainder = np.bincount(owner, weights=weights, minlength=n).astype(np.int64)
    values = (q << k) | remainder
    values[~is_plain] = escapes
    return values

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real-Time SC Bitstream Oscilloscope + HIL Debugger

"""Live bitstream oscilloscope for SC hardware debugging.

Analyses u32-packed bitstream samples as they arrive: density, popcount,
an entropy measure, stochastic-computing correlation, per-layer windows,
error budgets against expected densities and capture triggers.

Samples come from a transport backend. Only the simulated backend
produces data; the JTAG, UART and PYNQ DMA types are configuration
values whose backends report a connection and read nothing, so reading
from hardware needs a deployment-specific backend.

- **TransportBackend**: Pluggable adapters for JTAG, UART, PYNQ DMA, or
  simulated (loopback) bitstream sources.
- **BitstreamSample**: Timestamped bitstream capture with metadata.
- **LiveAnalyzer**: Windowed computation of density, popcount and the
  entropy measure per layer.
- **LayerErrorBudget**: Per-layer precision tracking against golden model.
- **TriggerEngine**: Conditional capture triggers on density.
- **ScopeSession**: Manages streaming, analysis, and trigger evaluation.
- **ScopeRenderer**: Text-mode (CLI) rendering of live scope data.

The module imports none of the tracer, analysis or profiling modules.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from sc_neurocore._native.array_guards import require_c_contiguous

try:
    from sc_neurocore.stochastic_doctor import stochastic_doctor_core as _sdc

    # `stochastic_doctor/__init__.py` now returns `None` on missing .so
    # rather than raising ImportError (so module import never hard-fails);
    # the real gate is whether the attribute is a loaded extension.
    _HAS_RUST_SCC = _sdc is not None
except ImportError:
    _sdc = None
    _HAS_RUST_SCC = False


# ── Transport Backends ───────────────────────────────────────────────


class TransportType(Enum):
    """The source a transport backend reads from.

    Only ``SIMULATED`` produces data in this module.
    """

    JTAG = "jtag"
    UART = "uart"
    PYNQ_DMA = "pynq_dma"
    SIMULATED = "simulated"


@dataclass
class TransportConfig:
    """Configuration for a transport backend.

    Attributes
    ----------
    transport_type : TransportType
        The source.
    port : str
        Port name for a serial or JTAG link.
    baud_rate : int
        Serial speed.
    dma_base_addr : int
        Base address of a DMA region.
    dma_length : int
        Length of the DMA region in bytes.
    timeout_ms : int
        Read timeout in milliseconds.

    Notes
    -----
    The backend of this module reads none of these fields; they are for a
    deployment-specific backend.
    """

    transport_type: TransportType
    port: str = ""
    baud_rate: int = 115200
    dma_base_addr: int = 0x4000_0000
    dma_length: int = 4096
    timeout_ms: int = 100


@dataclass
class TransportBackend:
    """Pluggable transport adapter for bitstream acquisition.

    Production backends (JTAG, UART, PYNQ DMA) require hardware;
    the ``SIMULATED`` backend generates synthetic data for testing
    and development. This class implements only the simulated one: for
    the other types it reports a connection and reads ``None``.

    Attributes
    ----------
    config : TransportConfig
        The backend's configuration.
    is_connected : bool
        Whether :meth:`connect` has been called since the last disconnect.
    bytes_received : int
        Bytes produced by simulated reads.
    """

    config: TransportConfig
    is_connected: bool = False
    bytes_received: int = 0
    _sim_rng: Optional[np.random.Generator] = field(default=None, repr=False)
    _sim_step: int = 0

    def connect(self) -> bool:
        """Establish connection to the target.

        The simulated backend seeds its generator with 42; the other types
        open nothing. Always returns ``True``.
        """
        if self.config.transport_type == TransportType.SIMULATED:
            self._sim_rng = np.random.default_rng(42)
            self.is_connected = True
            return True
        # Real backends would initialise JTAG/UART/DMA here
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Mark the backend disconnected and reset the simulation's state."""
        self.is_connected = False
        self._sim_rng = None
        self._sim_step = 0

    def read_bitstream(self, num_words: int, layer_id: int = 0) -> Optional[np.ndarray[Any, Any]]:
        """Read packed bitstream words from the target.

        Returns u32-packed words, or None on timeout/error.

        Without a connection, and for every type but ``SIMULATED``, the
        result is ``None``. A simulated read draws uniform 32-bit words and
        sets or clears only their top bit by a density of 0.3 + 0.1 ×
        ``layer_id`` with a slow sinusoidal change, clipped to [0.05, 0.95];
        the other 31 bits stay uniform, so the stream's bit density is close
        to 0.5 for every layer.
        """
        if not self.is_connected:
            return None

        if self.config.transport_type == TransportType.SIMULATED:
            return self._sim_read(num_words, layer_id)

        # Hardware transports are registered by deployment-specific backends.
        return None

    def _sim_read(self, num_words: int, layer_id: int) -> np.ndarray[Any, Any]:
        """Generate simulated bitstream data."""
        assert self._sim_rng is not None
        self._sim_step += 1

        # Simulate density that varies by layer and time
        base_density = 0.3 + 0.1 * layer_id
        time_mod = 0.1 * np.sin(self._sim_step * 0.05)
        density = np.clip(base_density + time_mod, 0.05, 0.95)

        threshold = int(density * 0xFFFF_FFFF)
        words = self._sim_rng.integers(0, 0xFFFF_FFFF, size=num_words, dtype=np.uint32)
        result = np.where(words < threshold, words | 0x8000_0000, words & 0x7FFF_FFFF)
        self.bytes_received += num_words * 4
        return result.astype(np.uint32)


# ── Bitstream Sample ─────────────────────────────────────────────────


@dataclass
class BitstreamSample:
    """One timestamped bitstream capture.

    Attributes
    ----------
    timestamp_ns : int
        Capture time in nanoseconds since the session started.
    layer_id : int
        Layer of the sample.
    neuron_id : int
        Neuron of the sample.
    words : numpy.ndarray
        The bitstream packed in 32-bit words.
    sample_index : int
        Position of the sample in its session.
    """

    timestamp_ns: int
    layer_id: int
    neuron_id: int
    words: np.ndarray[Any, Any]  # u32-packed bitstream
    sample_index: int = 0

    @property
    def bit_length(self) -> int:
        """Number of bits, 32 per word."""
        return len(self.words) * 32

    @property
    def popcount(self) -> int:
        """Number of one bits over all words."""
        total = 0
        for w in self.words:
            total += bin(int(w)).count("1")
        return total

    @property
    def density(self) -> float:
        """Fraction of one bits; 0.0 for an empty sample."""
        bl = self.bit_length
        return self.popcount / bl if bl > 0 else 0.0

    @property
    def effective_bits(self) -> float:
        """Shannon entropy-based effective precision.

        The binary entropy of the density times the bit length, H(d) × L:
        the information of L independent bits of that density, not the
        resolution of the encoded value. 0.0 at a density of 0 or 1.
        """
        p = self.density
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) * self.bit_length)


# ── Live Analyzer ────────────────────────────────────────────────────


@dataclass
class AnalysisWindow:
    """Windowed statistics from recent samples.

    Attributes
    ----------
    window_size : int
        Number of most recent samples kept.
    densities, popcounts, effective_bits, timestamps : collections.deque
        The kept samples' values, oldest first; rebuilt with
        ``window_size`` as their length at construction.
    """

    window_size: int = 64
    densities: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    popcounts: Deque[int] = field(default_factory=lambda: deque(maxlen=64))
    effective_bits: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    timestamps: Deque[int] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        """Replace the four buffers by empty ones of length ``window_size``."""
        self.densities = deque(maxlen=self.window_size)
        self.popcounts = deque(maxlen=self.window_size)
        self.effective_bits = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)

    def push(self, sample: BitstreamSample) -> None:
        """Add one sample's density, popcount, entropy measure and time."""
        self.densities.append(sample.density)
        self.popcounts.append(sample.popcount)
        self.effective_bits.append(sample.effective_bits)
        self.timestamps.append(sample.timestamp_ns)

    @property
    def count(self) -> int:
        """Number of samples in the window."""
        return len(self.densities)

    @property
    def mean_density(self) -> float:
        """Mean density of the window; 0.0 when empty."""
        return float(np.mean(self.densities)) if self.densities else 0.0

    @property
    def std_density(self) -> float:
        """Population standard deviation of the densities; 0.0 below two samples."""
        return float(np.std(self.densities)) if len(self.densities) > 1 else 0.0

    @property
    def mean_effective_bits(self) -> float:
        """Mean of the entropy measure; 0.0 when empty."""
        return float(np.mean(self.effective_bits)) if self.effective_bits else 0.0

    @property
    def total_popcount(self) -> int:
        """Sum of the window's popcounts."""
        return sum(self.popcounts)

    @property
    def sample_rate_hz(self) -> float:
        """Estimated sample rate from timestamps.

        Samples per second between the first and the last kept timestamp; 0.0
        below two samples or without elapsed time.
        """
        if len(self.timestamps) < 2:
            return 0.0
        dt_ns = self.timestamps[-1] - self.timestamps[0]
        if dt_ns <= 0:
            return 0.0
        return (len(self.timestamps) - 1) * 1e9 / dt_ns


def _compute_scc_python(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Pure-Python Alaghi-Hayes SCC (reference implementation + fallback)."""
    total_bits = len(a) * 32
    ones_a = sum(bin(int(w)).count("1") for w in a)
    ones_b = sum(bin(int(w)).count("1") for w in b)
    ones_ab = sum(bin(int(wa) & int(wb)).count("1") for wa, wb in zip(a, b))

    pa = ones_a / total_bits
    pb = ones_b / total_bits
    p_and = ones_ab / total_bits

    numerator = p_and - pa * pb
    if abs(numerator) < 1e-12:
        return 0.0
    if numerator > 0.0:
        # The AND density obeys p_and <= min(pa, pb), so this denominator is
        # always >= numerator > 1e-12 — no division-by-zero guard is reachable.
        denominator = min(pa, pb) - pa * pb
    else:
        # Symmetrically p_and >= max(0, pa + pb - 1), so denominator >= |numerator|.
        denominator = pa * pb - max(0.0, pa + pb - 1.0)
    return max(-1.0, min(1.0, numerator / denominator))


def compute_scc(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Stochastic Computing Correlation between two u32-packed bitstreams.

    Dispatches to the Rust ``stochastic_doctor_core.py_scc_packed`` when the
    compiled extension is importable (the default when the repo is built with
    ``maturin develop --release``). Falls back to :func:`_compute_scc_python`
    when the extension is missing — the fallback is numerically identical
    (both implement the case-split Alaghi & Hayes 2013 form).

    Parameters
    ----------
    a, b : numpy.ndarray
        The two bitstreams as 32-bit words; the native path needs them
        C-contiguous and reads two words as one 64-bit word, which assumes a
        little-endian host.

    Returns
    -------
    float
        The correlation in [-1, 1]; 0.0 for streams of different or zero
        length, or when the joint density equals the product of the
        densities.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    if _HAS_RUST_SCC:
        a32 = require_c_contiguous(a, "a", np.uint32)
        b32 = require_c_contiguous(b, "b", np.uint32)
        # Reinterpret pairs of u32 words as u64 for the Rust kernel. Popcount
        # is position-invariant inside a word, so viewing two adjacent u32s as
        # one u64 preserves the bit-level meaning on little-endian hosts. Pad
        # an odd-length array by one zero word.
        if a32.size % 2 == 1:
            a32 = np.concatenate([a32, np.zeros(1, dtype=np.uint32)])
            b32 = np.concatenate([b32, np.zeros(1, dtype=np.uint32)])
        a64 = a32.view(np.uint64)
        b64 = b32.view(np.uint64)
        total_bits = len(a) * 32
        return float(_sdc.py_scc_packed(a64, b64, total_bits))

    return _compute_scc_python(a, b)


class LiveAnalyzer:
    """Real-time SC bitstream analyzer with per-layer windows.

    Parameters
    ----------
    num_layers : int
        Layers whose windows exist from the start.
    window_size : int
        Window size of those layers.

    Attributes
    ----------
    num_layers : int
        The number given at construction.
    windows : dict of int to AnalysisWindow
        One window per layer seen; a layer first seen in a sample gets a
        window of 64 samples, not ``window_size``.
    total_samples : int
        Samples ingested.
    """

    def __init__(self, num_layers: int = 1, window_size: int = 64):
        self.num_layers = num_layers
        self.windows: Dict[int, AnalysisWindow] = {
            i: AnalysisWindow(window_size=window_size) for i in range(num_layers)
        }
        self.total_samples = 0

    def ingest(self, sample: BitstreamSample) -> None:
        """Process one incoming sample."""
        layer = sample.layer_id
        if layer not in self.windows:
            self.windows[layer] = AnalysisWindow()
        self.windows[layer].push(sample)
        self.total_samples += 1

    def layer_stats(self, layer_id: int) -> Dict[str, float]:
        """Get summary stats for one layer.

        Parameters
        ----------
        layer_id : int
            The layer.

        Returns
        -------
        dict of str to float
            Mean and standard deviation of density, mean entropy measure,
            total popcount, sample count and sample rate; empty for a layer
            without samples.
        """
        w = self.windows.get(layer_id)
        if w is None or w.count == 0:
            return {}
        return {
            "mean_density": w.mean_density,
            "std_density": w.std_density,
            "mean_effective_bits": w.mean_effective_bits,
            "total_popcount": w.total_popcount,
            "sample_count": w.count,
            "sample_rate_hz": w.sample_rate_hz,
        }

    def all_stats(self) -> Dict[int, Dict[str, float]]:
        """Return :meth:`layer_stats` of every layer with a window."""
        return {lid: self.layer_stats(lid) for lid in self.windows}


# ── Layer Error Budget ───────────────────────────────────────────────


@dataclass
class LayerErrorBudget:
    """Per-layer precision tracking against golden model expectations.

    Attributes
    ----------
    layer_id : int
        The layer.
    expected_density : float
        The golden model's density.
    tolerance : float
        Largest accepted absolute difference.
    history : list of float
        Every checked density, never trimmed.
    """

    layer_id: int
    expected_density: float
    tolerance: float = 0.05
    history: List[float] = field(default_factory=list)

    def check(self, measured_density: float) -> bool:
        """Check if measured density is within tolerance.

        The density is also appended to the history.
        """
        self.history.append(measured_density)
        return abs(measured_density - self.expected_density) <= self.tolerance

    @property
    def current_error(self) -> float:
        """Absolute error of the last checked density; 0.0 before any check."""
        if not self.history:
            return 0.0
        return abs(self.history[-1] - self.expected_density)

    @property
    def mean_error(self) -> float:
        """Mean absolute error over the history; 0.0 before any check."""
        if not self.history:
            return 0.0
        errors = [abs(h - self.expected_density) for h in self.history]
        return float(np.mean(errors))

    @property
    def max_error(self) -> float:
        """Largest absolute error over the history; 0.0 before any check."""
        if not self.history:
            return 0.0
        return max(abs(h - self.expected_density) for h in self.history)

    @property
    def violations(self) -> int:
        """Number of checked densities outside the tolerance."""
        return sum(1 for h in self.history if abs(h - self.expected_density) > self.tolerance)

    @property
    def pass_rate(self) -> float:
        """Fraction of checks within the tolerance; 1.0 before any check."""
        if not self.history:
            return 1.0
        return 1.0 - self.violations / len(self.history)


# ── Trigger Engine ───────────────────────────────────────────────────


class TriggerType(Enum):
    """The condition a trigger tests.

    :meth:`TriggerEngine.evaluate` tests ``DENSITY_ABOVE``, ``DENSITY_BELOW``
    and ``SPIKE_DETECTED`` (any one bit); ``SCC_ABOVE`` and
    ``ERROR_BUDGET_VIOLATION`` are accepted and never fire.
    """

    DENSITY_ABOVE = "density_above"
    DENSITY_BELOW = "density_below"
    SPIKE_DETECTED = "spike_detected"
    SCC_ABOVE = "scc_above"
    ERROR_BUDGET_VIOLATION = "error_violation"


@dataclass
class TriggerCondition:
    """Conditional capture trigger.

    Attributes
    ----------
    trigger_type : TriggerType
        What is tested.
    threshold : float
        The density limit of the density triggers.
    layer_id : int
        Only samples of this layer are tested.
    enabled : bool
        Whether the trigger is tested.
    """

    trigger_type: TriggerType
    threshold: float = 0.5
    layer_id: int = 0
    enabled: bool = True


@dataclass
class TriggerEvent:
    """A triggered capture event.

    Attributes
    ----------
    trigger_type : TriggerType
        The trigger that fired.
    timestamp_ns : int
        The sample's time.
    layer_id : int
        The sample's layer.
    measured_value : float
        The sample's density.
    threshold : float
        The trigger's threshold.
    sample : BitstreamSample
        The sample itself.
    """

    trigger_type: TriggerType
    timestamp_ns: int
    layer_id: int
    measured_value: float
    threshold: float
    sample: BitstreamSample


class TriggerEngine:
    """Evaluates capture triggers against incoming samples.

    Attributes
    ----------
    conditions : list of TriggerCondition
        The triggers, tested in order.
    events : list of TriggerEvent
        Fired events, the first ``max_events`` of them.
    max_events : int
        Largest number of events kept; later ones are returned but not kept.
    """

    def __init__(self) -> None:
        self.conditions: List[TriggerCondition] = []
        self.events: List[TriggerEvent] = []
        self.max_events: int = 1000

    def add_trigger(self, condition: TriggerCondition) -> None:
        """Append a trigger condition."""
        self.conditions.append(condition)

    def evaluate(self, sample: BitstreamSample) -> List[TriggerEvent]:
        """Check all triggers against a sample. Returns fired events.

        Each enabled trigger of the sample's layer is tested once: density
        above or below the threshold, or any one bit for a spike. The work
        grows with the number of triggers and of the sample's words.

        Parameters
        ----------
        sample : BitstreamSample
            The sample.

        Returns
        -------
        list of TriggerEvent
            The events of this sample, in trigger order.
        """
        fired = []
        for cond in self.conditions:
            if not cond.enabled:
                continue
            if cond.layer_id != sample.layer_id:
                continue

            triggered = False
            measured = 0.0
            if cond.trigger_type == TriggerType.DENSITY_ABOVE:
                measured = sample.density
                triggered = measured > cond.threshold
            elif cond.trigger_type == TriggerType.DENSITY_BELOW:
                measured = sample.density
                triggered = measured < cond.threshold
            elif cond.trigger_type == TriggerType.SPIKE_DETECTED:
                measured = sample.density
                triggered = measured > 0.0

            if triggered:
                event = TriggerEvent(
                    cond.trigger_type,
                    sample.timestamp_ns,
                    sample.layer_id,
                    measured,
                    cond.threshold,
                    sample,
                )
                fired.append(event)
                if len(self.events) < self.max_events:
                    self.events.append(event)
        return fired

    @property
    def event_count(self) -> int:
        """Number of kept events."""
        return len(self.events)

    def clear(self) -> None:
        """Drop every kept event; the triggers stay."""
        self.events.clear()


# ── Scope Session ────────────────────────────────────────────────────


@dataclass
class ScopeSession:
    """Manages a live debugging session.

    Attributes
    ----------
    transport : TransportBackend
        Where samples come from.
    analyzer : LiveAnalyzer
        Where samples are analysed.
    triggers : TriggerEngine
        The capture triggers.
    error_budgets : dict of int to LayerErrorBudget
        One budget per tracked layer.
    is_running : bool
        Whether the session has started and not stopped.
    sample_count : int
        Samples captured.
    """

    transport: TransportBackend
    analyzer: LiveAnalyzer
    triggers: TriggerEngine = field(default_factory=TriggerEngine)
    error_budgets: Dict[int, LayerErrorBudget] = field(default_factory=dict)
    is_running: bool = False
    sample_count: int = 0
    _start_time_ns: int = 0

    def start(self) -> bool:
        """Start the scope session.

        Connects the transport and records the start time.

        Returns
        -------
        bool
            ``False`` when the transport does not connect.
        """
        if not self.transport.connect():
            return False
        self.is_running = True
        self._start_time_ns = time.time_ns()
        return True

    def stop(self) -> None:
        """Stop the session and disconnect the transport; the start time stays."""
        self.is_running = False
        self.transport.disconnect()

    def add_error_budget(self, layer_id: int, expected_density: float, tol: float = 0.05) -> None:
        """Track one layer's density against an expected value.

        Parameters
        ----------
        layer_id : int
            The layer.
        expected_density : float
            The golden model's density.
        tol : float
            Largest accepted absolute difference.
        """
        self.error_budgets[layer_id] = LayerErrorBudget(layer_id, expected_density, tol)

    def capture_one(
        self, layer_id: int = 0, neuron_id: int = 0, num_words: int = 8
    ) -> Optional[BitstreamSample]:
        """Capture one bitstream sample from the target.

        The sample is analysed, checked against the layer's error budget if
        one exists, and tested by the triggers.

        Parameters
        ----------
        layer_id : int
            Layer to read.
        neuron_id : int
            Neuron recorded with the sample.
        num_words : int
            32-bit words to read.

        Returns
        -------
        BitstreamSample or None
            The sample; ``None`` when the session is not running or the
            transport read nothing.
        """
        if not self.is_running:
            return None
        words = self.transport.read_bitstream(num_words, layer_id)
        if words is None:
            return None
        ts = time.time_ns() - self._start_time_ns
        sample = BitstreamSample(
            timestamp_ns=ts,
            layer_id=layer_id,
            neuron_id=neuron_id,
            words=words,
            sample_index=self.sample_count,
        )
        self.sample_count += 1
        self.analyzer.ingest(sample)

        # Check error budgets
        if layer_id in self.error_budgets:
            self.error_budgets[layer_id].check(sample.density)

        # Evaluate triggers
        self.triggers.evaluate(sample)
        return sample

    def capture_sweep(self, num_layers: int, num_words: int = 8) -> List[BitstreamSample]:
        """Capture one sample from each layer.

        Parameters
        ----------
        num_layers : int
            Layers 0 to ``num_layers - 1`` are read, each as neuron 0.
        num_words : int
            32-bit words per sample.

        Returns
        -------
        list of BitstreamSample
            The samples that were read.
        """
        samples = []
        for lid in range(num_layers):
            s = self.capture_one(layer_id=lid, num_words=num_words)
            if s is not None:
                samples.append(s)
        return samples

    def status(self) -> Dict[str, Any]:
        """Return the session's state for display.

        Returns
        -------
        dict of str to Any
            Running flag, samples, seconds since the start (still counting
            after a stop), bytes received, kept trigger events and tracked
            layers.
        """
        elapsed = (time.time_ns() - self._start_time_ns) / 1e9 if self._start_time_ns else 0
        return {
            "running": self.is_running,
            "samples": self.sample_count,
            "elapsed_s": round(elapsed, 3),
            "bytes_received": self.transport.bytes_received,
            "triggers_fired": self.triggers.event_count,
            "layers_tracked": len(self.analyzer.windows),
        }


# ── Scope Renderer (CLI text-mode) ──────────────────────────────────


class ScopeRenderer:
    """Text-mode rendering of live scope data for CLI output."""

    BAR_WIDTH = 40

    @classmethod
    def render_density_bar(cls, density: float, width: int = 40) -> str:
        """Render a density as a text bar.

        A density outside [0, 1] is not clamped, so the bar is then longer
        than ``width``: above 1 with more filled cells, below 0 with more
        empty ones.
        """
        filled = int(density * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {density:.3f}"

    @classmethod
    def render_layer_summary(cls, layer_id: int, stats: Dict[str, float]) -> str:
        """Render one layer's statistics as one line.

        Parameters
        ----------
        layer_id : int
            The layer.
        stats : dict of str to float
            Its :meth:`LiveAnalyzer.layer_stats`.

        Returns
        -------
        str
            A density bar, the mean entropy measure and the sample count, or
            "(no data)" for empty statistics.
        """
        if not stats:
            return f"  L{layer_id}: (no data)"
        density = stats.get("mean_density", 0.0)
        eff = stats.get("mean_effective_bits", 0.0)
        n = int(stats.get("sample_count", 0))
        bar = cls.render_density_bar(density)
        return f"  L{layer_id}: {bar}  eff={eff:.1f}b  n={n}"

    @classmethod
    def render_session(cls, session: ScopeSession) -> str:
        """Render full session status as text.

        Error budgets show a check mark at a pass rate of 95 % or more.
        """
        lines = ["═══ SC Bitstream Scope ═══"]
        st = session.status()
        lines.append(f"  Status: {'● LIVE' if st['running'] else '○ STOPPED'}")
        lines.append(f"  Samples: {st['samples']}  Elapsed: {st['elapsed_s']}s")
        lines.append(f"  Bytes: {st['bytes_received']}  Triggers: {st['triggers_fired']}")
        lines.append("──────────────────────────")
        for lid in sorted(session.analyzer.windows.keys()):
            stats = session.analyzer.layer_stats(lid)
            lines.append(cls.render_layer_summary(lid, stats))
        if session.error_budgets:
            lines.append("── Error Budgets ────────")
            for lid, eb in sorted(session.error_budgets.items()):
                status = "✓" if eb.pass_rate >= 0.95 else "✗"
                lines.append(
                    f"  L{lid}: {status} err={eb.current_error:.4f} "
                    f"mean={eb.mean_error:.4f} pass={eb.pass_rate:.1%}"
                )
        return "\n".join(lines)

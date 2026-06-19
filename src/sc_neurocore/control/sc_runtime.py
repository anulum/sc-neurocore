# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Runtime Dynamic SC Adaptation Engine

"""On-device runtime for dynamic bitstream adaptation with built-in ECC.

Monitors activity statistics (popcount, SCC, drift) and dynamically
resizes bitstream lengths, switches decorrelators (LFSR → Sobol), or
injects Hamming(7,4) error-correcting codes without retraining.

The adaptation policy mirrors the Rust ``ScDoctor`` from
``dynamic_adaptation/src/lib.rs`` while extending it with decorrelator
switching and comprehensive logging.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DecorrelatorType(Enum):
    LFSR = "lfsr"
    SOBOL = "sobol"
    HALTON = "halton"
    HYBRID = "hybrid"


# Escalation order for decorrelator cascade
DECORRELATOR_CASCADE = [
    DecorrelatorType.LFSR,
    DecorrelatorType.SOBOL,
    DecorrelatorType.HALTON,
    DecorrelatorType.HYBRID,
]


class ECCMode(Enum):
    NONE = "none"
    PARITY = "parity"
    HAMMING = "hamming_7_4"
    SECDED = "secded_8_4"


class ActivityZone(Enum):
    IDLE = "idle"  # < 0.01
    LOW = "low"  # 0.01 – 0.05
    NORMAL = "normal"  # 0.05 – 0.5
    HIGH = "high"  # 0.5 – 0.95
    BURST = "burst"  # > 0.95


def classify_activity(density: float) -> ActivityZone:
    """Map popcount density to activity zone."""
    if density < 0.01:
        return ActivityZone.IDLE
    elif density < 0.05:
        return ActivityZone.LOW
    elif density <= 0.5:
        return ActivityZone.NORMAL
    elif density <= 0.95:
        return ActivityZone.HIGH
    else:
        return ActivityZone.BURST


@dataclass
class RuntimeConfig:
    """Current runtime configuration for a bitstream engine."""

    bitstream_length: int = 256
    decorrelator: DecorrelatorType = DecorrelatorType.LFSR
    ecc_enabled: bool = False
    ecc_mode: ECCMode = ECCMode.HAMMING
    ecc_overhead_bits: int = 0

    @property
    def effective_length(self) -> int:
        if self.ecc_enabled:
            if self.ecc_mode == ECCMode.SECDED:
                n_chunks = self.bitstream_length // 4
                return self.bitstream_length + n_chunks * 4  # 4 parity bits per 4 data
            elif self.ecc_mode == ECCMode.HAMMING:
                n_chunks = self.bitstream_length // 4
                return self.bitstream_length + n_chunks * 3
            elif self.ecc_mode == ECCMode.PARITY:
                n_chunks = self.bitstream_length // 8
                return self.bitstream_length + max(1, n_chunks)
        return self.bitstream_length

    def copy(self) -> RuntimeConfig:
        return RuntimeConfig(
            bitstream_length=self.bitstream_length,
            decorrelator=self.decorrelator,
            ecc_enabled=self.ecc_enabled,
            ecc_mode=self.ecc_mode,
            ecc_overhead_bits=self.ecc_overhead_bits,
        )


@dataclass
class AdaptationEvent:
    """Log entry for a runtime adaptation."""

    timestamp_ns: int
    trigger: str
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    metric_value: float


class ActivityMonitor:
    """Rolling-window bitstream activity tracker.

    Maintains running statistics of popcount density and SCC for
    drift detection.
    """

    def __init__(self, window_size: int = 100, drift_threshold: float = 0.3):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._density_history: deque[float] = deque(maxlen=window_size)
        self._scc_history: deque[float] = deque(maxlen=window_size)
        self._zone_history: deque[ActivityZone] = deque(maxlen=window_size)
        self._ema_scc: float = 0.0
        self._alpha: float = 0.1

    def observe(
        self,
        bitstream: np.ndarray[Any, Any],
        reference: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one observation.

        Returns dict with current metrics. Mixed value types — numeric
        fields plus `drift_detected: bool` and `activity_zone: str`, so
        the annotation is ``dict[str, Any]`` rather than
        ``dict[str, float]``.
        """
        density = float(np.mean(bitstream))
        self._density_history.append(density)

        zone = classify_activity(density)
        self._zone_history.append(zone)

        scc = 0.0
        if reference is not None and len(reference) == len(bitstream):
            scc = self._compute_scc(bitstream, reference)
        self._scc_history.append(scc)
        self._ema_scc = self._alpha * scc + (1 - self._alpha) * self._ema_scc

        return {
            "density": density,
            "scc": scc,
            "ema_scc": self._ema_scc,
            "drift_detected": abs(self._ema_scc) > self.drift_threshold,
            "mean_density": self.mean_density,
            "activity_zone": zone.value,
        }

    def _compute_scc(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        a_f = a.astype(np.float64).flatten()
        b_f = b.astype(np.float64).flatten()
        pa = float(np.mean(a_f))
        pb = float(np.mean(b_f))
        p_and = float(np.mean(a_f * b_f))
        num = p_and - pa * pb
        if abs(num) < 1e-12:
            return 0.0
        denom = (min(pa, pb) - pa * pb) if num > 0 else (pa * pb - max(0.0, pa + pb - 1.0))
        if abs(denom) < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, num / denom))

    @property
    def mean_density(self) -> float:
        return float(np.mean(list(self._density_history))) if self._density_history else 0.0

    @property
    def mean_scc(self) -> float:
        return float(np.mean(list(self._scc_history))) if self._scc_history else 0.0

    @property
    def drift_active(self) -> bool:
        return abs(self._ema_scc) > self.drift_threshold

    @property
    def current_zone(self) -> ActivityZone:
        return self._zone_history[-1] if self._zone_history else ActivityZone.NORMAL


class HammingECC:
    """Hamming(7,4) encoder/decoder (Python mirror of Rust ScDoctor ECC)."""

    @staticmethod
    def encode(data_4bit: int) -> int:
        """Encode 4-bit data to 7-bit Hamming codeword."""
        d1 = (data_4bit >> 3) & 1
        d2 = (data_4bit >> 2) & 1
        d3 = (data_4bit >> 1) & 1
        d4 = data_4bit & 1
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        return (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4

    @staticmethod
    def decode(encoded_7bit: int) -> int:
        """Decode 7-bit Hamming codeword to 4-bit data, correcting 1-bit errors."""
        p1 = (encoded_7bit >> 6) & 1
        p2 = (encoded_7bit >> 5) & 1
        d1 = (encoded_7bit >> 4) & 1
        p3 = (encoded_7bit >> 3) & 1
        d2 = (encoded_7bit >> 2) & 1
        d3 = (encoded_7bit >> 1) & 1
        d4 = encoded_7bit & 1

        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = (s3 << 2) | (s2 << 1) | s1

        corrected = encoded_7bit
        if syndrome > 0:
            bit_pos = [6, 5, 4, 3, 2, 1, 0]
            if syndrome <= 7:
                corrected ^= 1 << bit_pos[syndrome - 1]

        cd1 = (corrected >> 4) & 1
        cd2 = (corrected >> 2) & 1
        cd3 = (corrected >> 1) & 1
        cd4 = corrected & 1
        return (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4

    def encode_bitstream(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply Hamming(7,4) ECC to an entire bitstream (groups of 4 bits)."""
        n = len(bitstream)
        padded = np.zeros(((n + 3) // 4) * 4, dtype=np.uint8)
        padded[:n] = bitstream
        encoded = []
        for i in range(0, len(padded), 4):
            chunk = (
                (int(padded[i]) << 3)
                | (int(padded[i + 1]) << 2)
                | (int(padded[i + 2]) << 1)
                | int(padded[i + 3])
            )
            code = self.encode(chunk)
            for bit in range(6, -1, -1):
                encoded.append((code >> bit) & 1)
        return np.array(encoded, dtype=np.uint8)

    def decode_bitstream(self, encoded: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decode Hamming(7,4) encoded bitstream."""
        decoded = []
        for i in range(0, len(encoded) - 6, 7):
            code = 0
            for bit in range(7):
                code = (code << 1) | int(encoded[i + bit])
            data = self.decode(code)
            for bit in range(3, -1, -1):
                decoded.append((data >> bit) & 1)
        return np.array(decoded, dtype=np.uint8)


class SECDEC_ECC:
    """SECDED (Single Error Correct, Double Error Detect) Hamming(8,4).

    Extends Hamming(7,4) with an overall parity bit for 2-bit error
    detection.  Corrects all 1-bit errors, detects all 2-bit errors.
    """

    def __init__(self) -> None:
        self._hamming = HammingECC()

    def encode(self, data_4bit: int) -> int:
        """Encode 4-bit data to 8-bit SECDED codeword."""
        hamming_7 = self._hamming.encode(data_4bit)
        parity = bin(hamming_7).count("1") % 2
        return (parity << 7) | hamming_7

    def decode(self, encoded_8bit: int) -> Tuple[int, bool]:
        """Decode 8-bit SECDED codeword.

        Returns (data_4bit, uncorrectable).
        uncorrectable is True when a 2-bit error is detected.
        """
        overall_parity = (encoded_8bit >> 7) & 1
        hamming_7 = encoded_8bit & 0x7F

        # Compute syndrome
        p1 = (hamming_7 >> 6) & 1
        p2 = (hamming_7 >> 5) & 1
        d1 = (hamming_7 >> 4) & 1
        p3 = (hamming_7 >> 3) & 1
        d2 = (hamming_7 >> 2) & 1
        d3 = (hamming_7 >> 1) & 1
        d4 = hamming_7 & 1

        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = (s3 << 2) | (s2 << 1) | s1

        actual_parity = bin(encoded_8bit).count("1") % 2

        if syndrome == 0 and actual_parity == 0:
            # No error
            data = self._hamming.decode(hamming_7)
            return data, False
        elif syndrome != 0 and actual_parity != 0:
            # 1-bit error — correctable
            data = self._hamming.decode(hamming_7)
            return data, False
        elif syndrome != 0 and actual_parity == 0:
            # 2-bit error — uncorrectable, detected
            data = self._hamming.decode(hamming_7)
            return data, True
        else:
            # Parity bit itself is flipped — still correctable
            data = self._hamming.decode(hamming_7)
            return data, False

    def encode_bitstream(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply SECDED(8,4) to an entire bitstream."""
        n = len(bitstream)
        padded = np.zeros(((n + 3) // 4) * 4, dtype=np.uint8)
        padded[:n] = bitstream
        encoded = []
        for i in range(0, len(padded), 4):
            chunk = (
                (int(padded[i]) << 3)
                | (int(padded[i + 1]) << 2)
                | (int(padded[i + 2]) << 1)
                | int(padded[i + 3])
            )
            code = self.encode(chunk)
            for bit in range(7, -1, -1):
                encoded.append((code >> bit) & 1)
        return np.array(encoded, dtype=np.uint8)

    def decode_bitstream(self, encoded: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, Any], int]:
        """Decode SECDED(8,4) encoded bitstream.

        Returns (decoded_data, num_uncorrectable_errors).
        """
        decoded = []
        uncorrectable_count = 0
        for i in range(0, len(encoded) - 7, 8):
            code = 0
            for bit in range(8):
                code = (code << 1) | int(encoded[i + bit])
            data, uncorrectable = self.decode(code)
            if uncorrectable:
                uncorrectable_count += 1
            for bit in range(3, -1, -1):
                decoded.append((data >> bit) & 1)
        return np.array(decoded, dtype=np.uint8), uncorrectable_count


class AdaptationPolicy:
    """Rules for runtime bitstream adaptation."""

    def __init__(
        self,
        scc_high: float = 0.15,
        scc_low: float = 0.05,
        min_length: int = 256,
        max_length: int = 4096,
        ecc_trigger_length: int = 2048,
        enable_decorrelator_cascade: bool = True,
    ):
        self.scc_high = scc_high
        self.scc_low = scc_low
        self.min_length = min_length
        self.max_length = max_length
        self.ecc_trigger_length = ecc_trigger_length
        self.enable_cascade = enable_decorrelator_cascade

    def decide(
        self,
        config: RuntimeConfig,
        metrics: Dict[str, float],
    ) -> Tuple[RuntimeConfig, Optional[str]]:
        """Evaluate metrics and return (new_config, trigger_reason_or_None)."""
        new = config.copy()
        scc = abs(metrics.get("ema_scc", 0.0))
        drift = metrics.get("drift_detected", False)

        if scc > self.scc_high:
            new.bitstream_length = min(self.max_length, config.bitstream_length * 2)
            if new.bitstream_length > self.ecc_trigger_length:
                new.ecc_enabled = True
            return new, "high_scc"

        if scc < self.scc_low and config.bitstream_length > self.min_length:
            new.bitstream_length = max(self.min_length, config.bitstream_length // 2)
            new.ecc_enabled = False
            return new, "low_scc"

        if drift and self.enable_cascade:
            next_decorr = self._next_decorrelator(config.decorrelator)
            if next_decorr != config.decorrelator:
                new.decorrelator = next_decorr
                return new, "decorrelator_cascade"

        if drift and config.decorrelator == DecorrelatorType.LFSR:
            new.decorrelator = DecorrelatorType.SOBOL
            return new, "decorrelator_drift"

        return config, None

    @staticmethod
    def _next_decorrelator(current: DecorrelatorType) -> DecorrelatorType:
        """Get next decorrelator in escalation cascade."""
        try:
            idx = DECORRELATOR_CASCADE.index(current)
            if idx < len(DECORRELATOR_CASCADE) - 1:
                return DECORRELATOR_CASCADE[idx + 1]
        except ValueError:
            pass
        return current


@dataclass
class RuntimeReport:
    """Summary report from a runtime session."""

    total_observations: int = 0
    adaptations: List[AdaptationEvent] = field(default_factory=list)
    final_config: Optional[RuntimeConfig] = None
    uncorrectable_errors: int = 0

    @property
    def num_adaptations(self) -> int:
        return len(self.adaptations)

    def adaptation_rate(self, last_n: int = 0) -> float:
        """Adaptations per observation (optionally over last_n events)."""
        if self.total_observations == 0:
            return 0.0
        if last_n <= 0:
            return self.num_adaptations / self.total_observations
        recent = [e for e in self.adaptations[-last_n:]] if last_n else self.adaptations
        return len(recent) / max(1, min(last_n, self.total_observations))

    def summary(self) -> str:
        lines = [
            f"Runtime Report: {self.total_observations} observations, {self.num_adaptations} adaptations",
        ]
        if self.final_config:
            lines.append(
                f"  Final: length={self.final_config.bitstream_length}, "
                f"decorr={self.final_config.decorrelator.value}, "
                f"ecc={self.final_config.ecc_enabled} ({self.final_config.ecc_mode.value})"
            )
        if self.uncorrectable_errors > 0:
            lines.append(f"  Uncorrectable errors: {self.uncorrectable_errors}")
        return "\n".join(lines)


class SCRuntimeEngine:
    """Main runtime adapter: monitors + adapts + applies ECC."""

    def __init__(
        self,
        initial_config: Optional[RuntimeConfig] = None,
        policy: Optional[AdaptationPolicy] = None,
        monitor_window: int = 100,
    ):
        self.config = initial_config or RuntimeConfig()
        self.policy = policy or AdaptationPolicy()
        self.monitor = ActivityMonitor(window_size=monitor_window)
        self.ecc_hamming = HammingECC()
        self.ecc_secded = SECDEC_ECC()
        self.report = RuntimeReport(final_config=self.config)

    def observe(
        self,
        bitstream: np.ndarray[Any, Any],
        reference: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Feed one bitstream observation through the runtime.

        Returns current metrics and any adaptation triggered.
        """
        metrics = self.monitor.observe(bitstream, reference)
        self.report.total_observations += 1

        new_config, trigger = self.policy.decide(self.config, metrics)

        adapted = False
        if trigger is not None:
            event = AdaptationEvent(
                timestamp_ns=time.perf_counter_ns(),
                trigger=trigger,
                old_config={
                    "length": self.config.bitstream_length,
                    "decorrelator": self.config.decorrelator.value,
                    "ecc": self.config.ecc_enabled,
                    "ecc_mode": self.config.ecc_mode.value,
                },
                new_config={
                    "length": new_config.bitstream_length,
                    "decorrelator": new_config.decorrelator.value,
                    "ecc": new_config.ecc_enabled,
                    "ecc_mode": new_config.ecc_mode.value,
                },
                metric_value=metrics.get("ema_scc", 0.0),
            )
            self.report.adaptations.append(event)
            self.config = new_config
            self.report.final_config = new_config
            adapted = True

        return {
            **metrics,
            "adapted": adapted,
            "trigger": trigger,
            "config_length": self.config.bitstream_length,
            "config_ecc": self.config.ecc_enabled,
            "config_ecc_mode": self.config.ecc_mode.value,
        }

    def protect(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply ECC protection if enabled in current config."""
        if not self.config.ecc_enabled:
            return bitstream
        if self.config.ecc_mode == ECCMode.SECDED:
            return self.ecc_secded.encode_bitstream(bitstream)
        elif self.config.ecc_mode == ECCMode.HAMMING:
            return self.ecc_hamming.encode_bitstream(bitstream)
        elif self.config.ecc_mode == ECCMode.PARITY:
            # Simple even parity per 8-bit chunk
            n = len(bitstream)
            chunks = (n + 7) // 8
            padded = np.zeros(chunks * 8, dtype=np.uint8)
            padded[:n] = bitstream
            out: list[int] = []
            for i in range(0, len(padded), 8):
                chunk = padded[i : i + 8]
                out.extend(int(v) for v in chunk)
                out.append(int(np.sum(chunk) % 2))
            return np.array(out, dtype=np.uint8)
        return bitstream

    def recover(self, encoded: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decode ECC-protected bitstream if enabled."""
        if not self.config.ecc_enabled:
            return encoded
        if self.config.ecc_mode == ECCMode.SECDED:
            decoded, n_unc = self.ecc_secded.decode_bitstream(encoded)
            self.report.uncorrectable_errors += n_unc
            return decoded
        elif self.config.ecc_mode == ECCMode.HAMMING:
            return self.ecc_hamming.decode_bitstream(encoded)
        elif self.config.ecc_mode == ECCMode.PARITY:
            decoded_bits: list[int] = []
            for i in range(0, len(encoded) - 8, 9):
                decoded_bits.extend(int(v) for v in encoded[i : i + 8])
            return np.array(decoded_bits, dtype=np.uint8)
        return encoded

    def protect_batch(self, bitstreams: List[np.ndarray[Any, Any]]) -> List[np.ndarray[Any, Any]]:
        """Apply ECC to a batch of bitstreams."""
        return [self.protect(bs) for bs in bitstreams]

    def recover_batch(self, encoded_list: List[np.ndarray[Any, Any]]) -> List[np.ndarray[Any, Any]]:
        """Decode a batch of ECC-protected bitstreams."""
        return [self.recover(enc) for enc in encoded_list]

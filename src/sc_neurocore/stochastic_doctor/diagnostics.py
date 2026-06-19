# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Doctor Diagnostics
# Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

"""Bitstream-level stochastic correlation analysis and diagnostics.

Extends ``sc_neurocore.doctor`` with bitstream-specific metrics:

- **SCC**: Stochastic Cross-Correlation (Alaghi & Hayes, 2013)
- **Precision estimation**: empirical P̂ with variance bound σ² = p(1-p)/N
- **Drift detection**: EMA-based correlation drift monitoring
- **Activity histograms**: per-word popcount distribution

Includes optional Rust FFI acceleration via ``stochastic_doctor_core``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from sc_neurocore._native.array_guards import require_c_contiguous


# ---------------------------------------------------------------------------
# Rust Acceleration Configuration
# ---------------------------------------------------------------------------

import os as _os

_HAS_PYO3 = False
_sdc_rust = None

if not _os.environ.get("SC_NEUROCORE_NO_RUST"):
    try:
        from sc_neurocore.stochastic_doctor import stochastic_doctor_core as _sdc_rust

        # `stochastic_doctor/__init__.py` now returns `None` on missing
        # .so rather than raising — so the real gate is a non-None value.
        _HAS_PYO3 = _sdc_rust is not None
    except ImportError:
        pass

# Legacy ctypes fallback (secondary)
HAS_RUST_CORE = _HAS_PYO3
_libdoctor = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class AuditSeverity(Enum):
    """Audit finding severity levels."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class BitstreamAuditFinding:
    """Single finding from a bitstream audit."""

    category: str
    severity: AuditSeverity
    message: str
    metric: float = 0.0
    neuron_pair: Optional[tuple[int, int]] = None


@dataclass
class BitstreamAuditReport:
    """Full bitstream-level audit report for a network layer.

    JSON-serializable via ``to_json()`` for pipeline integration.
    """

    layer: str
    stream_length: int
    num_neurons: int
    max_correlation: float = 0.0
    mean_precision: float = 0.0
    precision_variance: float = 0.0
    hot_neurons: List[tuple[int, int, float]] = field(default_factory=list)
    findings: List[BitstreamAuditFinding] = field(default_factory=list)
    status: AuditSeverity = AuditSeverity.OK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        d = asdict(self)
        d["status"] = self.status.value
        d["findings"] = [{**asdict(f), "severity": f.severity.value} for f in self.findings]
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# SCC computation
# ---------------------------------------------------------------------------


def _scc_python(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Pure Python SCC (fallback when Rust core unavailable)."""
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    p_and = float(np.mean(np.bitwise_and(a, b)))
    numerator = p_and - (pa * pb)
    if abs(numerator) < 1e-12:
        return 0.0
    if numerator > 0:
        denominator = min(pa, pb) - (pa * pb)
    else:
        denominator = (pa * pb) - max(0.0, pa + pb - 1.0)
    if abs(denominator) < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def compute_scc(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Compute SCC between two bitstreams.

    Uses Rust PyO3 acceleration when available, falls back to pure Python.
    Set ``SC_NEUROCORE_NO_RUST=1`` to force Python path.
    """
    if _HAS_PYO3 and _sdc_rust is not None:
        a = require_c_contiguous(a, "a", np.uint8)
        b = require_c_contiguous(b, "b", np.uint8)
        return float(_sdc_rust.py_scc_bytes(a, b))
    a = np.ascontiguousarray(a, dtype=np.uint8)
    b = np.ascontiguousarray(b, dtype=np.uint8)
    return _scc_python(a, b)


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Exponential moving average drift detector for SCC monitoring.

    Tracks the running EMA of SCC values. Flags a drift event when
    |EMA| exceeds the threshold.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0.0–1.0; lower = smoother).
    threshold : float
        Absolute SCC value above which to flag drift.
    """

    def __init__(self, alpha: float = 0.1, threshold: float = 0.3):
        self.alpha = max(0.0, min(1.0, alpha))
        self.threshold = max(0.0, min(1.0, threshold))
        self.ema: float = 0.0
        self.active: bool = False
        self._history: List[float] = []

    def observe(self, scc_value: float) -> bool:
        """Feed a new SCC observation. Returns True if drift detected."""
        self.ema = self.alpha * scc_value + (1.0 - self.alpha) * self.ema
        self.active = abs(self.ema) > self.threshold
        self._history.append(self.ema)
        return self.active

    def reset(self) -> None:
        """Reset detector state."""
        self.ema = 0.0
        self.active = False
        self._history.clear()

    @property
    def history(self) -> List[float]:
        """EMA history for plotting/logging."""
        return self._history


# ---------------------------------------------------------------------------
# Main Doctor class
# ---------------------------------------------------------------------------


class StochasticDoctor:
    """Bitstream-level stochastic diagnostics engine.

    Parameters
    ----------
    correlation_threshold : float
        |SCC| above this triggers a WARNING (default 0.3).
    critical_threshold : float
        |SCC| above this triggers a CRITICAL (default 0.7).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        critical_threshold: float = 0.7,
    ):
        self.correlation_threshold = correlation_threshold
        self.critical_threshold = critical_threshold

    def compute_correlation(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Compute SCC between two bitstreams."""
        return compute_scc(a, b)

    def estimate_precision(self, bitstream: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Estimate probability and variance bound for a bitstream.

        Uses Rust PyO3 acceleration when available.

        Returns
        -------
        (probability, variance_bound)
        """
        if _HAS_PYO3 and _sdc_rust is not None:
            bs = require_c_contiguous(bitstream, "bitstream", np.uint8)
            return _sdc_rust.py_precision_bytes(bs)
        bs = np.ascontiguousarray(bitstream, dtype=np.uint8)
        n = len(bs)
        if n == 0:
            return (0.0, 0.0)
        p = float(np.mean(bs))
        variance = p * (1.0 - p) / n
        return (p, variance)

    def compute_histogram(
        self, bitstream: np.ndarray[Any, Any], word_size: int = 64
    ) -> np.ndarray[Any, Any]:
        """Compute per-word popcount histogram.

        Splits the bitstream into chunks of ``word_size`` and counts
        the popcount of each chunk. Returns a histogram with bins 0..word_size.
        Uses Rust PyO3 acceleration when available.

        Parameters
        ----------
        bitstream : ndarray
            1D array of 0/1 values.
        word_size : int
            Number of bits per word (default 64).
        """
        if _HAS_PYO3 and _sdc_rust is not None:
            bs = require_c_contiguous(bitstream, "bitstream", np.uint8)
            return np.asarray(_sdc_rust.py_histogram(bs, word_size))
        bs = np.ascontiguousarray(bitstream, dtype=np.uint8)
        n = len(bs)
        hist = np.zeros(word_size + 1, dtype=np.int64)
        for start in range(0, n, word_size):
            chunk = bs[start : start + word_size]
            pc = int(np.sum(chunk))
            hist[pc] += 1
        return hist

    def audit_layer(self, layer_id: str, bitstreams: np.ndarray[Any, Any]) -> BitstreamAuditReport:
        """Audit a full layer of bitstreams.

        Parameters
        ----------
        layer_id : str
            Human-readable layer identifier.
        bitstreams : ndarray
            Shape (num_neurons, stream_length), each element 0 or 1.

        Returns
        -------
        BitstreamAuditReport
        """
        num_neurons, stream_len = bitstreams.shape
        report = BitstreamAuditReport(
            layer=layer_id,
            stream_length=stream_len,
            num_neurons=num_neurons,
        )

        # Precision analysis
        precisions = []
        for i in range(num_neurons):
            p, var = self.estimate_precision(bitstreams[i])
            precisions.append(p)

        report.mean_precision = float(np.mean(precisions))
        report.precision_variance = float(np.var(precisions))

        # Pairwise SCC analysis
        max_corr = 0.0
        hot_pairs: List[tuple[int, int, float]] = []

        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                scc_val = self.compute_correlation(bitstreams[i], bitstreams[j])
                abs_scc = abs(scc_val)

                if abs_scc > abs(max_corr):
                    max_corr = scc_val

                if abs_scc > self.critical_threshold:
                    hot_pairs.append((i, j, scc_val))
                    report.findings.append(
                        BitstreamAuditFinding(
                            category="critical_correlation",
                            severity=AuditSeverity.CRITICAL,
                            message=f"Neurons ({i},{j}): SCC={scc_val:.4f} exceeds critical threshold",
                            metric=scc_val,
                            neuron_pair=(i, j),
                        )
                    )
                elif abs_scc > self.correlation_threshold:
                    hot_pairs.append((i, j, scc_val))
                    report.findings.append(
                        BitstreamAuditFinding(
                            category="high_correlation",
                            severity=AuditSeverity.WARNING,
                            message=f"Neurons ({i},{j}): SCC={scc_val:.4f} exceeds warning threshold",
                            metric=scc_val,
                            neuron_pair=(i, j),
                        )
                    )

        report.max_correlation = max_corr
        report.hot_neurons = hot_pairs

        # Overall status
        if any(f.severity == AuditSeverity.CRITICAL for f in report.findings):
            report.status = AuditSeverity.CRITICAL
        elif any(f.severity == AuditSeverity.WARNING for f in report.findings):
            report.status = AuditSeverity.WARNING
        else:
            report.status = AuditSeverity.OK

        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    doc = StochasticDoctor()

    s1 = np.zeros(2048, dtype=np.uint8)
    s1[:1024] = 1
    s2 = 1 - s1

    report = doc.audit_layer("V1_Cortex", np.stack([s1, s2]))
    print(report.to_json())

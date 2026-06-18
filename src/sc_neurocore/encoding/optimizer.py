# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automatic spike encoding optimizer

"""Auto-select optimal encoding scheme based on data characteristics.

Profiles input data (sparsity, temporal structure, dynamic range) and
scores each encoding scheme. Returns ranked recommendations.

No framework provides automatic encoding selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import encoders


@dataclass
class EncodingRecommendation:
    """Recommendation for one encoding scheme."""

    encoding: str
    score: float  # 0-1, higher is better
    sparsity: float  # fraction of zeros in encoded output
    info_preserved: float  # estimated information preservation
    reason: str


class EncodingOptimizer:
    """Profile data and recommend optimal spike encoding.

    Parameters
    ----------
    T : int
        Number of simulation timesteps.
    """

    def __init__(self, T: int = 32):
        self.T = T

    def profile(self, data: np.ndarray[Any, Any]) -> dict[str, float]:
        """Compute data statistics relevant to encoding selection.

        Parameters
        ----------
        data : ndarray of shape (N,) or (T_data, N)
            Input data (values should be in [0, 1] or will be normalized).

        Returns
        -------
        dict with: mean, std, sparsity, temporal_autocorrelation, dynamic_range
        """
        d = data.astype(np.float64)
        if d.max() > 1.0 or d.min() < 0.0:
            d = (d - d.min()) / max(d.max() - d.min(), 1e-8)

        stats = {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "sparsity": float(np.mean(d < 0.01)),
            "dynamic_range": float(d.max() - d.min()),
        }

        if d.ndim == 2 and d.shape[0] > 1:
            autocorr = (
                float(
                    np.mean(
                        [
                            np.corrcoef(d[:-1, i], d[1:, i])[0, 1]
                            for i in range(d.shape[1])
                            if np.std(d[:, i]) > 1e-8
                        ]
                    )
                )
                if d.shape[1] > 0
                else 0.0
            )
            stats["temporal_autocorrelation"] = autocorr
        else:
            stats["temporal_autocorrelation"] = 0.0

        return stats

    def recommend(self, data: np.ndarray[Any, Any]) -> list[EncodingRecommendation]:
        """Recommend encoding schemes ranked by suitability.

        Parameters
        ----------
        data : ndarray of shape (N,) or (T_data, N)

        Returns
        -------
        list of EncodingRecommendation, sorted by score descending
        """
        stats = self.profile(data)
        recs = []

        # Normalize data to [0, 1] for encoding
        d = data.astype(np.float64).ravel() if data.ndim == 1 else data.astype(np.float64)
        if d.max() > 1.0 or d.min() < 0.0:
            d = (d - d.min()) / max(d.max() - d.min(), 1e-8)

        sample = d[:100] if d.ndim == 1 else d[0, :100] if d.ndim == 2 else d.ravel()[:100]

        # Score each encoding
        for name, enc_fn, score_fn in self._encodings():
            encoded = enc_fn(sample, self.T) if name != "delta" and name != "sigma_delta" else None
            if encoded is not None:
                sparsity = float(1.0 - encoded.mean())
                info = self._info_score(sample, encoded)
            else:  # pragma: no cover
                sparsity = 0.5
                info = 0.5

            base_score = score_fn(stats)
            final_score = 0.5 * base_score + 0.3 * info + 0.2 * (0.5 + 0.5 * sparsity)

            recs.append(
                EncodingRecommendation(
                    encoding=name,
                    score=float(np.clip(final_score, 0, 1)),
                    sparsity=sparsity,
                    info_preserved=info,
                    reason=self._reason(name, stats),
                )
            )

        recs.sort(key=lambda r: r.score, reverse=True)
        return recs

    def _info_score(self, original: np.ndarray[Any, Any], encoded: np.ndarray[Any, Any]) -> float:
        """Estimate how well encoding preserves input information."""
        decoded_approx = encoded.mean(axis=0)
        if len(decoded_approx) != len(original):  # pragma: no cover
            return 0.5
        corr = np.corrcoef(original, decoded_approx)[0, 1]
        return float(max(0, corr)) if np.isfinite(corr) else 0.0

    def _encodings(self) -> list[tuple[str, Any, Any]]:
        return [
            ("rate", encoders.rate_encode, lambda s: 0.7 + 0.3 * (1 - s["sparsity"])),
            ("latency", encoders.latency_encode, lambda s: 0.8 if s["sparsity"] < 0.5 else 0.4),
            (
                "phase",
                encoders.phase_encode,
                lambda s: 0.6 + 0.3 * s.get("temporal_autocorrelation", 0),
            ),
            ("burst", encoders.burst_encode, lambda s: 0.5 + 0.3 * s["dynamic_range"]),
            ("rank_order", encoders.rank_order_encode, lambda s: 0.7 if s["std"] > 0.2 else 0.3),
        ]

    def _reason(self, name: str, stats: dict[str, float]) -> str:
        reasons = {
            "rate": "Good general-purpose encoding, works well with diverse data",
            "latency": "Low-latency single-spike encoding, energy-efficient",
            "phase": "Captures periodic structure in temporal data",
            "burst": "Preserves intensity information in burst length",
            "rank_order": "Exploits relative ordering, good for high-variance data",
        }
        return reasons.get(name, "")

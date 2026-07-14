# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic crosstalk analysis

"""Coupled-mode crosstalk models with an optional Rust execution path."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple, cast

import numpy as np

from ._photonic_types import _require_finite, _require_non_negative, _require_positive

CrosstalkAnalyzer = Callable[..., Mapping[str, object]]


def _unavailable_analyzer(**_kwargs: object) -> Mapping[str, object]:
    """Raise for an unavailable optional Rust analyzer."""
    raise ImportError("Rust photonic crosstalk backend unavailable")


try:
    from sc_neurocore_engine.photonics import (
        get_crosstalk_analyzer,
        get_crosstalk_bank_analyzer,
        get_crosstalk_pair_analyzer,
        has_full_photonic_crosstalk_backend,
    )
except ImportError:

    def get_crosstalk_analyzer() -> object:
        """Return the optional Rust scalar crosstalk analyzer or raise."""
        raise ImportError("Rust photonic crosstalk backend unavailable")

    def get_crosstalk_bank_analyzer() -> object:
        """Return the optional Rust crosstalk-bank analyzer or raise."""
        raise ImportError("Rust photonic crosstalk bank backend unavailable")

    def get_crosstalk_pair_analyzer() -> object:
        """Return the optional Rust crosstalk-pair analyzer or raise."""
        raise ImportError("Rust photonic crosstalk pair backend unavailable")

    def has_full_photonic_crosstalk_backend() -> bool:
        """Return whether all optional Rust crosstalk kernels are available."""
        return False


py_ph_analyze_crosstalk: CrosstalkAnalyzer = _unavailable_analyzer
py_ph_analyze_crosstalk_bank: CrosstalkAnalyzer = _unavailable_analyzer
py_ph_analyze_crosstalk_pairs: CrosstalkAnalyzer = _unavailable_analyzer
try:
    py_ph_analyze_crosstalk = cast(CrosstalkAnalyzer, get_crosstalk_analyzer())
    py_ph_analyze_crosstalk_bank = cast(CrosstalkAnalyzer, get_crosstalk_bank_analyzer())
    py_ph_analyze_crosstalk_pairs = cast(CrosstalkAnalyzer, get_crosstalk_pair_analyzer())
    _HAS_RUST_PH = has_full_photonic_crosstalk_backend()
except ImportError:
    _HAS_RUST_PH = False


def _facade_binding(name: str, default: object) -> object:
    """Read a compatibility binding from the historical facade when loaded."""
    facade = sys.modules.get("sc_neurocore.optics.photonic_emitter")
    return getattr(facade, name, default) if facade is not None else default


def _rust_backend_enabled() -> bool:
    """Return the effective compatibility-controlled Rust backend flag."""
    return bool(_facade_binding("_HAS_RUST_PH", _HAS_RUST_PH))


def _backend_analyzer(name: str, default: CrosstalkAnalyzer) -> CrosstalkAnalyzer:
    """Return a dynamically replaceable compatibility analyzer."""
    return cast(CrosstalkAnalyzer, _facade_binding(name, default))


def _require_index(value: int, name: str) -> None:
    """Reject a Boolean, non-integer, or negative waveguide index."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass
class WaveguidePair:
    """Physical contract for one pair of adjacent optical waveguides."""

    waveguide_width_nm: float = 450.0
    gap_nm: float = 200.0
    coupling_length_um: float = 10.0
    core_index: float = 3.48
    cladding_index: float = 1.45
    wavelength_nm: float = 1550.0

    def __post_init__(self) -> None:
        """Validate the coupled-mode domain before numerical evaluation."""
        _require_positive(self.waveguide_width_nm, "waveguide_width_nm")
        _require_non_negative(self.gap_nm, "gap_nm")
        _require_non_negative(self.coupling_length_um, "coupling_length_um")
        _require_positive(self.core_index, "core_index")
        _require_positive(self.cladding_index, "cladding_index")
        if self.core_index <= self.cladding_index:
            raise ValueError("core_index must be greater than cladding_index")
        _require_positive(self.wavelength_nm, "wavelength_nm")

    @property
    def effective_index_diff(self) -> float:
        """Return the Marcatili-form even/odd effective-index difference."""
        decay_length_nm = self.wavelength_nm / (
            2 * math.pi * math.sqrt(self.core_index**2 - self.cladding_index**2)
        )
        return 0.1 * math.exp(-self.gap_nm / decay_length_nm)

    @property
    def coupling_coefficient(self) -> float:
        """Return coupling coefficient κ per micrometre."""
        return math.pi * self.effective_index_diff / (self.wavelength_nm * 1e-3)

    @property
    def coupling_ratio(self) -> float:
        """Return power coupling ratio at the end of the parallel run."""
        kl = self.coupling_coefficient * self.coupling_length_um
        return math.sin(kl) ** 2

    @property
    def isolation_db(self) -> float:
        """Return pair isolation in decibels with a 300 dB numeric ceiling."""
        ratio = self.coupling_ratio
        if ratio < 1e-15:
            return 300.0
        return -10.0 * math.log10(max(ratio, 1e-30))


class CrosstalkModel:
    """Evaluate evanescent crosstalk between parallel waveguide runs."""

    def __init__(self) -> None:
        self.pairs: List[WaveguidePair] = []

    def add_pair(self, pair: WaveguidePair) -> None:
        """Append one validated waveguide pair to the analyzer batch."""
        if not isinstance(pair, WaveguidePair):
            raise TypeError("pair must be a WaveguidePair")
        self.pairs.append(pair)

    def transfer_matrix(self, pair: WaveguidePair) -> np.ndarray[Any, Any]:
        """Return the two-by-two unitary directional-coupler matrix."""
        if not isinstance(pair, WaveguidePair):
            raise TypeError("pair must be a WaveguidePair")
        kl = pair.coupling_coefficient * pair.coupling_length_um
        cosine = math.cos(kl)
        sine = math.sin(kl)
        return np.array([[cosine, 1j * sine], [1j * sine, cosine]])

    def compute_crosstalk(
        self, pair: WaveguidePair, input_power: Tuple[float, float] = (1.0, 0.0)
    ) -> Tuple[float, float]:
        """Return output power on both waveguides for two input amplitudes."""
        if len(input_power) != 2:
            raise ValueError("input_power must contain exactly two field amplitudes")
        _require_finite(input_power[0], "input_power[0]")
        _require_finite(input_power[1], "input_power[1]")
        transfer = self.transfer_matrix(pair)
        output = transfer @ np.array(input_power, dtype=complex)
        return float(np.abs(output[0]) ** 2), float(np.abs(output[1]) ** 2)

    def worst_case_isolation(self) -> float:
        """Return minimum isolation across registered pairs in decibels."""
        if not self.pairs:
            return float("inf")
        return min(pair.isolation_db for pair in self.pairs)

    def analyze_bank(
        self,
        waveguides: int,
        gap_nm: float,
        coupling_length_um: float,
        wavelength_nm: float = 1550.0,
        core_index: float = 3.48,
        cladding_index: float = 1.45,
    ) -> Dict[str, Any]:
        """Analyze adjacent and next-nearest pairs in a uniform waveguide bank."""
        _require_index(waveguides, "waveguides")
        if waveguides < 1:
            raise ValueError("waveguides must be >= 1")
        near = WaveguidePair(
            gap_nm=gap_nm,
            coupling_length_um=coupling_length_um,
            wavelength_nm=wavelength_nm,
            core_index=core_index,
            cladding_index=cladding_index,
        )
        far = WaveguidePair(
            gap_nm=2.0 * gap_nm,
            coupling_length_um=coupling_length_um,
            wavelength_nm=wavelength_nm,
            core_index=core_index,
            cladding_index=cladding_index,
        )

        if _rust_backend_enabled():
            analyzer = _backend_analyzer(
                "py_ph_analyze_crosstalk_bank", py_ph_analyze_crosstalk_bank
            )
            return dict(
                analyzer(
                    num_waveguides=waveguides,
                    gap_nm=gap_nm,
                    coupling_length_um=coupling_length_um,
                    wavelength_nm=wavelength_nm,
                    core_index=core_index,
                    cladding_index=cladding_index,
                )
            )

        num_near = max(0, waveguides - 1)
        num_far = max(0, waveguides - 2)
        total = num_near + num_far
        if total == 0:
            worst = float("inf")
            mean_ratio = 0.0
            max_ratio = 0.0
        else:
            worst = min(near.isolation_db, far.isolation_db)
            mean_ratio = (num_near * near.coupling_ratio + num_far * far.coupling_ratio) / total
            max_ratio = max(near.coupling_ratio, far.coupling_ratio)
        return {
            "num_waveguides": waveguides,
            "num_pairs": total,
            "num_near_pairs": num_near,
            "num_far_pairs": num_far,
            "gap_nm": gap_nm,
            "coupling_length_um": coupling_length_um,
            "adjacent_coupling_ratio": near.coupling_ratio,
            "adjacent_isolation_db": near.isolation_db,
            "next_nearest_coupling_ratio": far.coupling_ratio,
            "next_nearest_isolation_db": far.isolation_db,
            "worst_isolation_db": worst,
            "mean_coupling_ratio": mean_ratio,
            "max_coupling_ratio": max_ratio,
            "crosstalk_safe": worst > 20.0,
            "backend": "python",
        }

    def analyze_pairs(
        self,
        pair_indices: List[Tuple[int, int]],
        gaps_nm: List[float],
        coupling_lengths_um: List[float],
        wavelength_nm: float = 1550.0,
        core_index: float = 3.48,
        cladding_index: float = 1.45,
    ) -> Dict[str, Any]:
        """Analyze per-pair crosstalk for arbitrary waveguide geometry."""
        pair_count = len(pair_indices)
        if len(gaps_nm) != pair_count or len(coupling_lengths_um) != pair_count:
            raise ValueError(
                f"pair_indices ({pair_count}), gaps_nm ({len(gaps_nm)}) and "
                f"coupling_lengths_um ({len(coupling_lengths_um)}) must be equal length"
            )
        for index, (pair_a, pair_b) in enumerate(pair_indices):
            _require_index(pair_a, f"pair_indices[{index}][0]")
            _require_index(pair_b, f"pair_indices[{index}][1]")
            if pair_a == pair_b:
                raise ValueError(f"pair_indices[{index}] must name two distinct waveguides")

        pairs = [
            WaveguidePair(
                gap_nm=gap,
                coupling_length_um=length,
                wavelength_nm=wavelength_nm,
                core_index=core_index,
                cladding_index=cladding_index,
            )
            for gap, length in zip(gaps_nm, coupling_lengths_um)
        ]
        if _rust_backend_enabled() and pair_count > 0:
            analyzer = _backend_analyzer(
                "py_ph_analyze_crosstalk_pairs", py_ph_analyze_crosstalk_pairs
            )
            return dict(
                analyzer(
                    pairs_a=[pair_a for pair_a, _ in pair_indices],
                    pairs_b=[pair_b for _, pair_b in pair_indices],
                    gaps_nm=list(gaps_nm),
                    lengths_um=list(coupling_lengths_um),
                    wavelength_nm=wavelength_nm,
                    core_index=core_index,
                    cladding_index=cladding_index,
                )
            )

        return {
            "pair_a": [pair_a for pair_a, _ in pair_indices],
            "pair_b": [pair_b for _, pair_b in pair_indices],
            "gap_nm": list(gaps_nm),
            "coupling_length_um": list(coupling_lengths_um),
            "coupling_coefficient_per_um": [pair.coupling_coefficient for pair in pairs],
            "coupling_ratio": [pair.coupling_ratio for pair in pairs],
            "isolation_db": [pair.isolation_db for pair in pairs],
            "num_pairs": pair_count,
            "backend": "python",
        }


__all__ = [
    "CrosstalkModel",
    "WaveguidePair",
    "_HAS_RUST_PH",
    "py_ph_analyze_crosstalk",
    "py_ph_analyze_crosstalk_bank",
    "py_ph_analyze_crosstalk_pairs",
]

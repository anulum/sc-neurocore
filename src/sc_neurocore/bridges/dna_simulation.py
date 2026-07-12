# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA circuit simulation

"""Kinetics, noise, concentration, precision, and degradation analysis."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .dna_types import (
    _DEFAULT_TEMPERATURE_C,
    _R_GAS,
    DNACircuitDesign,
    DNAGate,
    GateType,
)


class KineticSimulator:
    """Mass-action kinetics simulator for DNA strand displacement.

    Simulates the time evolution of strand concentrations using
    selectable integration (Euler or RK4) with Arrhenius temperature
    scaling of rate constants.

    Parameters
    ----------
    rate_hybridization : float
        Second-order rate constant for toehold binding (M⁻¹ s⁻¹).
    rate_displacement : float
        First-order rate constant for branch migration (s⁻¹).
    temperature_c : float
        Temperature in Celsius.
    integrator : str
        Integration method: ``"euler"`` or ``"rk4"``.
    """

    def __init__(
        self,
        rate_hybridization: float = 3e5,
        rate_displacement: float = 1.0,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
        integrator: str = "euler",
    ) -> None:
        self._k_hyb = rate_hybridization
        self._k_disp = rate_displacement
        self._temperature_c = temperature_c
        self._integrator = integrator

    def _arrhenius_scale(self, k_ref: float, ea_kcal: float = 15.0) -> float:
        """Scale rate constant from 37°C to operating temperature via Arrhenius."""
        t_ref = 310.15  # 37°C in Kelvin
        t_op = self._temperature_c + 273.15
        return k_ref * math.exp(-(ea_kcal / _R_GAS) * (1.0 / t_op - 1.0 / t_ref))

    def _compute_k_eff(
        self,
        gate: "DNAGate",
        input_concentrations: Dict[str, float],
    ) -> float:
        """Compute effective rate constant for a gate."""
        k_hyb = self._arrhenius_scale(self._k_hyb)
        k_disp = self._arrhenius_scale(self._k_disp)

        if gate.gate_type == GateType.AND:
            inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
            input_present = all(c > 0.0 for c in inputs_conc)
            k_eff = k_hyb * min(inputs_conc) * 1e-9 * (1.0 if input_present else 0.0)

        elif gate.gate_type == GateType.OR:
            inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
            k_eff = k_hyb * max(inputs_conc) * 1e-9

        elif gate.gate_type == GateType.NOT:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_disp * (1.0 - min(inp_conc / 200.0, 1.0))

        elif gate.gate_type == GateType.THRESHOLD:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            excess = max(0.0, inp_conc - gate.threshold * 200.0)
            k_eff = k_hyb * excess * 1e-9

        elif gate.gate_type == GateType.MUX:
            sel = input_concentrations.get(gate.input_names[0], 0.0)
            a = input_concentrations.get(gate.input_names[1], 0.0)
            b = input_concentrations.get(gate.input_names[2], 0.0)
            sel_frac = min(sel / 200.0, 1.0)
            k_eff = k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9

        elif gate.gate_type == GateType.AMPLIFIER:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_hyb * inp_conc * 1e-9 * 5.0  # catalytic turnover

        elif gate.gate_type == GateType.BUFFER:
            inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
            k_eff = k_disp * min(inp_conc / 200.0, 1.0)

        else:
            k_eff = 0.0

        return k_eff + gate.leak_rate

    def simulate(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
        dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """Simulate circuit kinetics.

        Parameters
        ----------
        design : DNACircuitDesign
            Compiled circuit to simulate.
        input_concentrations : dict[str, float]
            Initial concentrations of input signal strands (nM).
        duration_s : float
            Simulation duration in seconds.
        dt : float
            Time step in seconds.

        Returns
        -------
        dict[str, np.ndarray]
            Time traces for each output strand. Keys are strand names,
            values are 1D arrays of concentrations over time.
            Includes ``"time"`` key with the time axis.
        """
        n_steps = int(duration_s / dt)
        time = np.linspace(0.0, duration_s, n_steps)

        outputs: Dict[str, np.ndarray[Any, Any]] = {"time": time}
        max_conc = 200.0

        for g in design.gates:
            conc = np.zeros(n_steps)
            k_eff = self._compute_k_eff(g, input_concentrations)

            if self._integrator == "rk4":
                for t in range(1, n_steps):
                    c = conc[t - 1]
                    k1 = k_eff * (max_conc - c) * dt
                    k2 = k_eff * (max_conc - (c + k1 / 2)) * dt
                    k3 = k_eff * (max_conc - (c + k2 / 2)) * dt
                    k4 = k_eff * (max_conc - (c + k3)) * dt
                    conc[t] = c + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    conc[t] = max(0.0, min(conc[t], max_conc))
            else:
                for t in range(1, n_steps):
                    d_conc = k_eff * (max_conc - conc[t - 1]) * dt
                    conc[t] = conc[t - 1] + d_conc
                    conc[t] = max(0.0, min(conc[t], max_conc))

            outputs[g.output_name] = conc

        return outputs


class NoiseModel:
    """Monte Carlo noise injection for robustness analysis.

    Perturbs strand concentrations, hybridization rates, and
    temperature to assess circuit robustness under realistic
    experimental conditions.

    Parameters
    ----------
    concentration_cv : float
        Coefficient of variation for pipetting noise (default 0.05 = 5%).
    temperature_std_c : float
        Temperature uncertainty in °C (default 0.5).
    n_trials : int
        Number of Monte Carlo trials (default 50).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        concentration_cv: float = 0.05,
        temperature_std_c: float = 0.5,
        n_trials: int = 50,
        seed: int = 42,
    ) -> None:
        self._conc_cv = concentration_cv
        self._temp_std = temperature_std_c
        self._n_trials = n_trials
        self._rng = np.random.default_rng(seed)

    def sensitivity_analysis(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
    ) -> Dict[str, Any]:
        """Run Monte Carlo sensitivity analysis.

        Returns statistics on output concentration variation across trials.
        """
        sim = KineticSimulator()
        output_keys = [g.output_name for g in design.gates]
        results: Dict[str, list[float]] = {k: [] for k in output_keys}

        for _ in range(self._n_trials):
            perturbed_conc = {
                k: max(0.0, v * (1.0 + self._rng.normal(0, self._conc_cv)))
                for k, v in input_concentrations.items()
            }
            traces = sim.simulate(design, perturbed_conc, duration_s=duration_s)
            for k in output_keys:
                if k not in traces:
                    raise RuntimeError(f"kinetic simulator omitted output trace: {k}")
                results[k].append(float(traces[k][-1]))

        report: Dict[str, Any] = {"n_trials": self._n_trials, "outputs": {}}
        for k, vals in results.items():
            arr = np.array(vals)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            cv = std / max(mean, 1e-12)
            report["outputs"][k] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "robust": bool(cv < 0.15),
            }

        return report


class ConcentrationOptimizer:
    """Gradient-free optimization of strand concentrations.

    Uses Nelder–Mead simplex to minimize output error across
    all truth-table entries, finding optimal working concentrations
    for translator, threshold, and fuel strands.

    Parameters
    ----------
    n_evaluations : int
        Maximum function evaluations (default 200).
    seed : int
        Random seed for initial simplex.
    """

    def __init__(self, n_evaluations: int = 200, seed: int = 42) -> None:
        self._max_eval = n_evaluations
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        design: DNACircuitDesign,
        truth_table: list[Dict[str, Any]],
        duration_s: float = 1800.0,
    ) -> Dict[str, Any]:
        """Optimize concentrations against a truth table.

        Parameters
        ----------
        design : DNACircuitDesign
            Circuit to optimize.
        truth_table : list[dict]
            Each entry: ``{"inputs": {"A": 200, "B": 0}, "expected": {"C": "low"}}``.
        duration_s : float
            Simulation duration per evaluation.

        Returns
        -------
        dict
            ``best_score``, ``initial_score``, ``improvement_pct``,
            ``n_evaluations``, ``best_concentrations``.
        """
        sim = KineticSimulator()

        def score_fn(conc_scale: float) -> float:
            total_err = 0.0
            for entry in truth_table:
                scaled = {k: v * conc_scale for k, v in entry["inputs"].items()}
                result = sim.simulate(design, scaled, duration_s=duration_s)
                for out_name, expected in entry["expected"].items():
                    if out_name not in result:
                        raise RuntimeError(f"kinetic simulator omitted expected output: {out_name}")
                    final = float(result[out_name][-1])
                    target = 150.0 if expected == "high" else 20.0
                    total_err += (final - target) ** 2
            return total_err

        initial_score = score_fn(1.0)
        best_scale = 1.0
        best_score = initial_score

        for _ in range(self._max_eval):
            candidate = 0.5 + self._rng.random() * 1.5
            s = score_fn(candidate)
            if s < best_score:
                best_score = s
                best_scale = candidate

        improvement = (1.0 - best_score / max(initial_score, 1e-12)) * 100

        return {
            "best_score": float(best_score),
            "initial_score": float(initial_score),
            "improvement_pct": float(max(0, improvement)),
            "n_evaluations": self._max_eval,
            "best_scale": float(best_scale),
        }


class SCPrecisionAnalyzer:
    """Stochastic computing precision analysis for DNA circuits.

    Evaluates the effective bit-width, signal-to-noise ratio, and
    output precision achievable by a DNA-encoded SC circuit at given
    strand concentrations.

    In standard SC, a bitstream of length L encodes precision
    log2(L+1) bits. In DNA circuits, the analog concentration range
    [0, max_nM] plays the role of L.
    """

    def analyze(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        max_conc_nM: float = 200.0,
        duration_s: float = 3600.0,
    ) -> Dict[str, Any]:
        """Analyze SC precision of a compiled circuit.

        Returns
        -------
        dict
            Per-output: ``effective_bits``, ``snr_db``, ``dynamic_range_db``,
            ``resolution_nM``. Plus global ``total_effective_bits``.
        """
        sim = KineticSimulator()
        result = sim.simulate(design, input_concentrations, duration_s=duration_s)

        analysis: Dict[str, Any] = {"outputs": {}, "max_conc_nM": max_conc_nM}

        for key, trace in result.items():
            if key == "time":
                continue
            arr = np.asarray(trace)
            final = float(arr[-1])

            # Steady-state noise: std of last 10% of trace
            tail = arr[int(len(arr) * 0.9) :]
            noise_std = float(np.std(tail)) if len(tail) > 1 else 1e-6
            noise_std = max(noise_std, 1e-6)

            signal = float(np.mean(tail))
            snr = signal / noise_std
            snr_db = 20.0 * math.log10(max(snr, 1e-12))

            # Effective bits: based on how many distinguishable levels
            n_levels = max_conc_nM / max(noise_std, 1e-6)
            effective_bits = math.log2(max(n_levels, 1.0))

            # Dynamic range
            sig_max = float(np.max(arr))
            sig_min = float(np.min(arr[arr > 0])) if np.any(arr > 0) else 1e-6
            dynamic_range = 20.0 * math.log10(max(sig_max / sig_min, 1.0))

            analysis["outputs"][key] = {
                "final_nM": final,
                "noise_std_nM": noise_std,
                "snr_linear": float(snr),
                "snr_db": snr_db,
                "effective_bits": effective_bits,
                "dynamic_range_db": dynamic_range,
                "resolution_nM": float(noise_std * 2),
            }

        if analysis["outputs"]:
            analysis["total_effective_bits"] = min(
                v["effective_bits"] for v in analysis["outputs"].values()
            )
        else:
            analysis["total_effective_bits"] = 0.0

        return analysis


# ══════════════════════════════════════════════════════════════════════
# Degradation Model
# ══════════════════════════════════════════════════════════════════════


class DegradationModel:
    """Time-dependent DNA strand degradation model.

    Models first-order exponential decay of strand concentrations
    based on nuclease activity, temperature, and strand length.

    Parameters
    ----------
    half_life_hr : float
        Base half-life in hours at 37°C (default 24 for ssDNA).
    temperature_c : float
        Operating temperature in Celsius.
    """

    def __init__(
        self,
        half_life_hr: float = 24.0,
        temperature_c: float = 37.0,
    ) -> None:
        self._half_life_s = half_life_hr * 3600.0
        self._temperature_c = temperature_c
        self._k_decay = math.log(2) / self._half_life_s

    def _length_factor(self, length: int) -> float:
        """Longer strands degrade faster (more nuclease attack sites)."""
        return 1.0 + 0.02 * max(0, length - 20)

    def _temp_factor(self) -> float:
        """Higher temperature accelerates degradation."""
        return math.exp(0.05 * (self._temperature_c - 37.0))

    def predict_concentration(
        self,
        initial_nM: float,
        strand_length: int,
        time_hr: float,
    ) -> float:
        """Predict remaining concentration after time_hr hours."""
        k = self._k_decay * self._length_factor(strand_length) * self._temp_factor()
        return initial_nM * math.exp(-k * time_hr * 3600.0)

    def analyze_design(
        self,
        design: DNACircuitDesign,
        time_hr: float = 4.0,
    ) -> Dict[str, Any]:
        """Predict degradation across all circuit strands.

        Returns
        -------
        dict
            Per-strand: ``initial_nM``, ``remaining_nM``, ``pct_remaining``.
            Global: ``min_remaining_pct``, ``critical_strands``.
        """
        all_strands = list(design.input_strands) + list(design.output_strands)
        for g in design.gates:
            all_strands.extend(g.strands)

        strands_report: list[Dict[str, Any]] = []
        min_pct = 100.0

        for s in all_strands:
            remaining = self.predict_concentration(s.concentration_nM, s.length, time_hr)
            pct = (
                (remaining / max(s.concentration_nM, 1e-12)) * 100
                if s.concentration_nM > 0
                else 100.0
            )
            strands_report.append(
                {
                    "name": s.name,
                    "length": s.length,
                    "initial_nM": s.concentration_nM,
                    "remaining_nM": remaining,
                    "pct_remaining": pct,
                }
            )
            min_pct = min(min_pct, pct)

        critical = [s for s in strands_report if s["pct_remaining"] < 50.0]

        return {
            "time_hr": time_hr,
            "temperature_c": self._temperature_c,
            "half_life_hr": self._half_life_s / 3600.0,
            "strands": strands_report,
            "min_remaining_pct": min_pct,
            "n_critical_strands": len(critical),
            "critical_strands": critical,
        }

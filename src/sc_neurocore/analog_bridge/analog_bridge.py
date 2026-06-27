# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-to-Analog Bridge

"""DAC/ADC bridge for hybrid stochastic-analog computing.

Supports event-driven AER interfaces, analog substrate profiles
for BrainScaleS-3 and DynapSE, and on-chip calibration routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class AnalogSubstrateProfile:
    """Parameter set for analog/mixed-signal neuromorphic chips."""

    name: str
    g_min: float  # minimum conductance (nS)
    g_max: float  # maximum conductance (nS)
    v_min: float  # minimum membrane voltage (mV)
    v_max: float  # maximum membrane voltage (mV)
    dac_resolution: int  # bits
    tau_mem_range: Tuple[float, float] = (1.0, 100.0)  # membrane time constant (ms)
    tau_syn_range: Tuple[float, float] = (0.5, 50.0)  # synaptic time constant (ms)
    max_fanin: int = 256

    @classmethod
    def brainscales3(cls) -> AnalogSubstrateProfile:
        """Return the bundled BrainScaleS-3 substrate profile."""
        return cls(
            name="BrainScaleS-3",
            g_min=0.0,
            g_max=63.0,
            v_min=-80.0,
            v_max=-40.0,
            dac_resolution=6,
            tau_mem_range=(1.0, 50.0),
            tau_syn_range=(0.5, 20.0),
            max_fanin=256,
        )

    @classmethod
    def dynapse2(cls) -> AnalogSubstrateProfile:
        """Return the bundled DynapSE-2 substrate profile."""
        return cls(
            name="DynapSE-2",
            g_min=0.0,
            g_max=127.0,
            v_min=-70.0,
            v_max=-30.0,
            dac_resolution=7,
            tau_mem_range=(5.0, 200.0),
            tau_syn_range=(1.0, 100.0),
            max_fanin=64,
        )


@dataclass
class AEREvent:
    """Address-Event Representation spike event."""

    neuron_id: int
    timestamp_us: float
    polarity: int = 1  # 1 = excitatory, -1 = inhibitory


class AnalogBridge:
    """Quantize stochastic weights and thresholds for analog substrates."""

    def __init__(
        self,
        g_range: Tuple[float, float] | None = None,
        v_range: Tuple[float, float] | None = None,
        dac_res: int = 10,
        profile: AnalogSubstrateProfile | None = None,
    ):
        if profile is not None:
            self.g_min, self.g_max = profile.g_min, profile.g_max
            self.v_min, self.v_max = profile.v_min, profile.v_max
            self.dac_res = profile.dac_resolution
            self.profile: AnalogSubstrateProfile | None = profile
        else:
            self.g_min, self.g_max = g_range or (0.0, 100.0)
            self.v_min, self.v_max = v_range or (-80.0, -40.0)
            self.dac_res = dac_res
            self.profile = None
        self.dac_levels = 2**self.dac_res

    def _quantize(self, val: float, v_min: float, v_max: float) -> Tuple[int, float]:
        """Return ``(dac_value, actual_analog_value)`` after quantization."""
        norm = (val - v_min) / (v_max - v_min)
        norm = max(0.0, min(1.0, norm))
        dac = int(round(norm * (self.dac_levels - 1)))
        actual = v_min + (dac / (self.dac_levels - 1)) * (v_max - v_min)
        return dac, actual

    def emit_analog_config(self, nodes: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Emit DAC configuration dictionaries for SC weights and LIF nodes."""
        config: Dict[str, Dict[str, Any]] = {"synapses": {}, "neurons": {}, "errors": {}}
        for n in nodes:
            if n.type == "SC_WEIGHT":
                target_g = self.g_min + n.probability * (self.g_max - self.g_min)
                dac, actual = self._quantize(target_g, self.g_min, self.g_max)
                config["synapses"][n.id] = {"dac": dac, "g_ns": actual}
                config["errors"][n.id] = abs(target_g - actual)
            elif n.type == "LIF_MEMBRANE":
                target_v = self.v_min + n.threshold * (self.v_max - self.v_min)
                dac, actual = self._quantize(target_v, self.v_min, self.v_max)
                config["neurons"][n.id] = {"dac": dac, "v_mv": actual}
        return config


class EventDrivenInterface:
    """Converts between SC bitstreams and AER event streams."""

    def __init__(self, clock_period_us: float = 1.0):
        self.clock_period_us = clock_period_us

    def bitstream_to_events(
        self, neuron_id: int, bitstream: np.ndarray[Any, Any]
    ) -> List[AEREvent]:
        """Convert a boolean bitstream to a sequence of AER spike events."""
        events = []
        for i, bit in enumerate(bitstream):
            if bit:
                events.append(
                    AEREvent(
                        neuron_id=neuron_id,
                        timestamp_us=i * self.clock_period_us,
                    )
                )
        return events

    def events_to_current(
        self,
        events: List[AEREvent],
        duration_us: float,
        tau_syn: float = 5.0,
        weight: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Convert AER events to time-discretized synaptic current trace.

        Applies an exponential decay kernel per event.
        """
        n_steps = max(1, int(duration_us / self.clock_period_us))
        current = np.zeros(n_steps)
        for ev in events:
            idx = int(ev.timestamp_us / self.clock_period_us)
            if 0 <= idx < n_steps:
                for t in range(idx, n_steps):
                    dt = (t - idx) * self.clock_period_us
                    current[t] += weight * ev.polarity * np.exp(-dt / tau_syn)
        return current

    def rate_code(self, events: List[AEREvent], window_us: float) -> float:
        """Compute firing rate (Hz) from an event list."""
        if not events or window_us <= 0:
            return 0.0
        return len(events) / (window_us * 1e-6)


class CalibrationRoutine:
    """On-chip characterization loop for analog substrate alignment."""

    def __init__(self, bridge: AnalogBridge, num_steps: int = 10):
        self.bridge = bridge
        self.num_steps = num_steps

    def sweep_conductance(self) -> List[Tuple[int, float, float]]:
        """Sweep DAC range and report (dac_value, target_g, actual_g) tuples."""
        results = []
        for step in range(self.num_steps + 1):
            frac = step / self.num_steps
            target = self.bridge.g_min + frac * (self.bridge.g_max - self.bridge.g_min)
            dac, actual = self.bridge._quantize(target, self.bridge.g_min, self.bridge.g_max)
            results.append((dac, target, actual))
        return results

    def max_quantization_error(self) -> float:
        """Return worst-case quantization error across the conductance range."""
        sweep = self.sweep_conductance()
        return max(abs(target - actual) for _, target, actual in sweep)

    def effective_resolution_bits(self) -> float:
        """Compute effective number of bits (ENOB) given quantization errors."""
        max_err = self.max_quantization_error()
        full_range = self.bridge.g_max - self.bridge.g_min
        if max_err == 0 or full_range == 0:
            return float(self.bridge.dac_res)
        return float(np.log2(full_range / max_err))


if __name__ == "__main__":
    # Demo with BrainScaleS-3 profile
    profile = AnalogSubstrateProfile.brainscales3()
    bridge = AnalogBridge(profile=profile)

    class MockNode:
        """Minimal node descriptor for the command-line demonstration."""

        def __init__(self, t: str, i: str, prob: float = 0.0, th: float = 0.0) -> None:
            self.type, self.id, self.probability, self.threshold = t, i, prob, th

    nodes: List[MockNode] = [
        MockNode("SC_WEIGHT", "s1", prob=0.33),
        MockNode("LIF_MEMBRANE", "n1", th=0.55),
    ]
    config = bridge.emit_analog_config(nodes)
    import json

    print(f"--- {profile.name} Analog Configuration ---")
    print(json.dumps(config, indent=2))

    # Calibration
    cal = CalibrationRoutine(bridge)
    print(f"\nMax Quantization Error: {cal.max_quantization_error():.4f} nS")
    print(f"Effective Resolution: {cal.effective_resolution_bits():.2f} bits")

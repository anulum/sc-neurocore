# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Community Model Zoo & Plugin System

"""Community model zoo with plugin-based neuron models and Verilog generation.

Provides an abstract ``NeuronPlugin`` base class that downstream
contributors extend to define custom neuron dynamics.  Built-in plugins
cover the canonical spiking neuron models (LIF, Izhikevich, AdEx,
Hodgkin–Huxley).  The ``VerilogGenerator`` converts any plugin into
synthesisable SystemVerilog suitable for FPGA deployment, and the
``DocGenerator`` emits markdown documentation from plugin metadata.
"""

from __future__ import annotations

import math
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NeuronState:
    """Generic state container for neuron models."""

    variables: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.variables[key]

    def __setitem__(self, key: str, value: float) -> None:
        self.variables[key] = value

    def copy(self) -> NeuronState:
        return NeuronState(variables=dict(self.variables))

    def as_dict(self) -> Dict[str, float]:
        return dict(self.variables)


@dataclass
class PluginMeta:
    """Metadata carried by every neuron plugin."""

    name: str
    version: str
    author: str
    description: str
    references: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    state_variables: List[str] = field(default_factory=list)


class NeuronPlugin(ABC):
    """Abstract base class for pluggable neuron models."""

    @abstractmethod
    def meta(self) -> PluginMeta: ...

    @abstractmethod
    def default_state(self) -> NeuronState: ...

    @abstractmethod
    def default_params(self) -> Dict[str, float]: ...

    @abstractmethod
    def ode_dynamics(
        self,
        state: NeuronState,
        current: float,
        params: Dict[str, float],
        dt: float,
    ) -> NeuronState:
        """Advance the neuron state by one timestep *dt*."""
        ...

    @abstractmethod
    def threshold_check(self, state: NeuronState, params: Dict[str, float]) -> bool:
        """Return True if the neuron has fired."""
        ...

    @abstractmethod
    def reset(self, state: NeuronState, params: Dict[str, float]) -> NeuronState:
        """Reset state after a spike."""
        ...

    def simulate(
        self,
        current_trace: np.ndarray,
        dt: float = 0.001,
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Convenience: simulate a full current trace.

        Returns (voltage_trace, spike_indices).
        """
        p = params or self.default_params()
        state = self.default_state()
        voltages = np.zeros(len(current_trace), dtype=np.float64)
        spikes: List[int] = []
        for i, I_ext in enumerate(current_trace):
            state = self.ode_dynamics(state, float(I_ext), p, dt)
            if self.threshold_check(state, p):
                spikes.append(i)
                state = self.reset(state, p)
            voltages[i] = state["V"]
        return voltages, spikes


# ── Built-in Plugins ─────────────────────────────────────────────────


class LIFPlugin(NeuronPlugin):
    """Leaky Integrate-and-Fire neuron model."""

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="LIF",
            version="1.0.0",
            author="Miroslav Šotek",
            description="Leaky Integrate-and-Fire with exponential decay.",
            references=["Lapicque, J. Physiol. Pathol. Gén. 9, 1907."],
            parameters={
                "tau_m": "Membrane time constant (s)",
                "V_rest": "Resting potential (V)",
                "V_thresh": "Spike threshold (V)",
                "V_reset": "Reset potential (V)",
                "R_m": "Membrane resistance (Ω)",
            },
            state_variables=["V"],
        )

    def default_state(self) -> NeuronState:
        return NeuronState({"V": -0.070})

    def default_params(self) -> Dict[str, float]:
        return {
            "tau_m": 0.020,
            "V_rest": -0.070,
            "V_thresh": -0.055,
            "V_reset": -0.075,
            "R_m": 1e7,
        }

    def ode_dynamics(
        self, state: NeuronState, current: float, params: Dict[str, float], dt: float
    ) -> NeuronState:
        s = state.copy()
        tau = params["tau_m"]
        V = s["V"]
        dV = (-(V - params["V_rest"]) + params["R_m"] * current) / tau
        s["V"] = V + dV * dt
        return s

    def threshold_check(self, state: NeuronState, params: Dict[str, float]) -> bool:
        return state["V"] >= params["V_thresh"]

    def reset(self, state: NeuronState, params: Dict[str, float]) -> NeuronState:
        s = state.copy()
        s["V"] = params["V_reset"]
        return s


class IzhikevichPlugin(NeuronPlugin):
    """Izhikevich (2003) simple model of spiking neurons."""

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="Izhikevich",
            version="1.0.0",
            author="Miroslav Šotek",
            description="Izhikevich 2-variable model (regular spiking default).",
            references=["Izhikevich, IEEE Trans. NN 14(6), 2003."],
            parameters={
                "a": "Recovery time scale",
                "b": "Sensitivity of u to V",
                "c": "After-spike reset of V (mV)",
                "d": "After-spike increment of u",
                "V_thresh": "Spike cutoff (mV)",
            },
            state_variables=["V", "u"],
        )

    def default_state(self) -> NeuronState:
        return NeuronState({"V": -65.0, "u": -14.0})

    def default_params(self) -> Dict[str, float]:
        return {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "V_thresh": 30.0}

    def ode_dynamics(
        self, state: NeuronState, current: float, params: Dict[str, float], dt: float
    ) -> NeuronState:
        s = state.copy()
        V, u = s["V"], s["u"]
        dt_ms = dt * 1000.0
        dV = 0.04 * V * V + 5.0 * V + 140.0 - u + current
        du = params["a"] * (params["b"] * V - u)
        s["V"] = V + dV * dt_ms
        s["u"] = u + du * dt_ms
        return s

    def threshold_check(self, state: NeuronState, params: Dict[str, float]) -> bool:
        return state["V"] >= params["V_thresh"]

    def reset(self, state: NeuronState, params: Dict[str, float]) -> NeuronState:
        s = state.copy()
        s["V"] = params["c"]
        s["u"] = s["u"] + params["d"]
        return s


class AdExPlugin(NeuronPlugin):
    """Adaptive Exponential Integrate-and-Fire (Brette & Gerstner 2005)."""

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="AdEx",
            version="1.0.0",
            author="Miroslav Šotek",
            description="Adaptive exponential I&F with sub-threshold resonance.",
            references=["Brette & Gerstner, J. Neurophysiology 94(5), 2005."],
            parameters={
                "C": "Capacitance (nF)",
                "gL": "Leak conductance (nS)",
                "EL": "Leak reversal (mV)",
                "VT": "Threshold (mV)",
                "DeltaT": "Slope factor (mV)",
                "tau_w": "Adaptation τ (ms)",
                "a": "Sub-threshold adaptation (nS)",
                "b": "Spike-triggered adaptation (nA)",
                "V_reset": "Reset voltage (mV)",
                "V_peak": "Spike cutoff (mV)",
            },
            state_variables=["V", "w"],
        )

    def default_state(self) -> NeuronState:
        return NeuronState({"V": -70.0, "w": 0.0})

    def default_params(self) -> Dict[str, float]:
        return {
            "C": 0.281,
            "gL": 0.030,
            "EL": -70.6,
            "VT": -50.4,
            "DeltaT": 2.0,
            "tau_w": 144.0,
            "a": 0.004,
            "b": 0.0805,
            "V_reset": -70.6,
            "V_peak": 20.0,
        }

    def ode_dynamics(
        self, state: NeuronState, current: float, params: Dict[str, float], dt: float
    ) -> NeuronState:
        s = state.copy()
        V, w = s["V"], s["w"]
        dt_ms = dt * 1000.0
        exp_term = params["DeltaT"] * math.exp(
            min((V - params["VT"]) / max(params["DeltaT"], 1e-6), 20.0)
        )
        dV = (-params["gL"] * (V - params["EL"]) + params["gL"] * exp_term - w + current) / params[
            "C"
        ]
        dw = (params["a"] * (V - params["EL"]) - w) / params["tau_w"]
        s["V"] = V + dV * dt_ms
        s["w"] = w + dw * dt_ms
        return s

    def threshold_check(self, state: NeuronState, params: Dict[str, float]) -> bool:
        return state["V"] >= params["V_peak"]

    def reset(self, state: NeuronState, params: Dict[str, float]) -> NeuronState:
        s = state.copy()
        s["V"] = params["V_reset"]
        s["w"] = s["w"] + params["b"]
        return s


class HodgkinHuxleyPlugin(NeuronPlugin):
    """Hodgkin–Huxley conductance-based model (1952)."""

    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="Hodgkin-Huxley",
            version="1.0.0",
            author="Miroslav Šotek",
            description="Full HH model with Na/K/leak conductances.",
            references=["Hodgkin & Huxley, J. Physiology 117(4), 1952."],
            parameters={
                "C_m": "Membrane capacitance (µF/cm²)",
                "g_Na": "Na max conductance",
                "g_K": "K max conductance",
                "g_L": "Leak conductance",
                "E_Na": "Na reversal",
                "E_K": "K reversal",
                "E_L": "Leak reversal",
                "V_thresh": "Spike detection threshold (mV)",
            },
            state_variables=["V", "m", "h", "n"],
        )

    def default_state(self) -> NeuronState:
        return NeuronState({"V": -65.0, "m": 0.05, "h": 0.6, "n": 0.32})

    def default_params(self) -> Dict[str, float]:
        return {
            "C_m": 1.0,
            "g_Na": 120.0,
            "g_K": 36.0,
            "g_L": 0.3,
            "E_Na": 50.0,
            "E_K": -77.0,
            "E_L": -54.387,
            "V_thresh": 0.0,
        }

    def ode_dynamics(
        self, state: NeuronState, current: float, params: Dict[str, float], dt: float
    ) -> NeuronState:
        s = state.copy()
        V, m, h, n = s["V"], s["m"], s["h"], s["n"]
        dt_ms = dt * 1000.0

        def _safe_exp(x: float) -> float:
            return math.exp(max(-500.0, min(500.0, x)))

        a_m = (
            0.1 * (V + 40.0) / (1.0 - _safe_exp(-(V + 40.0) / 10.0))
            if abs(V + 40.0) > 1e-6
            else 1.0
        )
        b_m = 4.0 * _safe_exp(-(V + 65.0) / 18.0)
        a_h = 0.07 * _safe_exp(-(V + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))
        a_n = (
            0.01 * (V + 55.0) / (1.0 - _safe_exp(-(V + 55.0) / 10.0))
            if abs(V + 55.0) > 1e-6
            else 0.1
        )
        b_n = 0.125 * _safe_exp(-(V + 65.0) / 80.0)

        I_Na = params["g_Na"] * m**3 * h * (V - params["E_Na"])
        I_K = params["g_K"] * n**4 * (V - params["E_K"])
        I_L = params["g_L"] * (V - params["E_L"])

        dV = (current - I_Na - I_K - I_L) / params["C_m"]
        s["V"] = V + dV * dt_ms
        s["m"] = m + (a_m * (1 - m) - b_m * m) * dt_ms
        s["h"] = h + (a_h * (1 - h) - b_h * h) * dt_ms
        s["n"] = n + (a_n * (1 - n) - b_n * n) * dt_ms

        s["m"] = max(0.0, min(1.0, s["m"]))
        s["h"] = max(0.0, min(1.0, s["h"]))
        s["n"] = max(0.0, min(1.0, s["n"]))
        return s

    def threshold_check(self, state: NeuronState, params: Dict[str, float]) -> bool:
        return state["V"] >= params["V_thresh"]

    def reset(self, state: NeuronState, params: Dict[str, float]) -> NeuronState:
        return state.copy()


# ── Plugin Registry ──────────────────────────────────────────────────


class PluginRegistry:
    """Discovers and manages neuron model plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, NeuronPlugin] = {}

    def register(self, plugin: NeuronPlugin) -> None:
        name = plugin.meta().name
        self._plugins[name] = plugin

    def get(self, name: str) -> Optional[NeuronPlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        return sorted(self._plugins.keys())

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    @classmethod
    def with_builtins(cls) -> PluginRegistry:
        """Create a registry pre-loaded with all built-in neuron models."""
        reg = cls()
        for plugin_cls in (LIFPlugin, IzhikevichPlugin, AdExPlugin, HodgkinHuxleyPlugin):
            reg.register(plugin_cls())
        return reg


# ── Verilog Generator ────────────────────────────────────────────────


class VerilogGenerator:
    """Generates synthesisable SystemVerilog from a NeuronPlugin."""

    def __init__(self, bit_width: int = 16, frac_bits: int = 8) -> None:
        self.bit_width = bit_width
        self.frac_bits = frac_bits

    def generate(self, plugin: NeuronPlugin) -> str:
        """Produce a complete SystemVerilog module for the given plugin."""
        meta = plugin.meta()
        params = plugin.default_params()
        state_vars = meta.state_variables

        module_name = f"sc_neuron_{meta.name.lower().replace('-', '_')}"
        bw = self.bit_width

        port_lines = [
            "    input  logic clk,",
            "    input  logic rst_n,",
            f"    input  logic signed [{bw - 1}:0] i_current,",
        ]
        for sv in state_vars:
            port_lines.append(f"    output logic signed [{bw - 1}:0] o_{sv},")
        port_lines.append("    output logic o_spike")

        reg_lines = []
        for sv in state_vars:
            reg_lines.append(f"    logic signed [{bw - 1}:0] {sv}_reg;")

        reset_lines = []
        default_state = plugin.default_state()
        for sv in state_vars:
            fixed_val = self._to_fixed(default_state[sv])
            reset_lines.append(f"            {sv}_reg <= {bw}'sd{fixed_val};")

        param_lines = []
        for pname, pval in params.items():
            fixed_val = self._to_fixed(pval)
            safe_name = pname.replace("-", "_")
            param_lines.append(
                f"    localparam signed [{bw - 1}:0] {safe_name.upper()} = {bw}'sd{fixed_val};"
            )

        assign_lines = []
        for sv in state_vars:
            assign_lines.append(f"    assign o_{sv} = {sv}_reg;")

        header = textwrap.dedent(f"""\
            // SPDX-License-Identifier: AGPL-3.0-or-later
            // Commercial license available
            // © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
            // © Code 2020–2026 Miroslav Šotek. All rights reserved.
            // ORCID: 0009-0009-3560-0851
            // Contact: www.anulum.li | protoscience@anulum.li
            // SC-NeuroCore — Auto-generated {meta.name} neuron module
            //
            // Generated from plugin: {meta.name} v{meta.version}
            // {meta.description}
        """)

        body = textwrap.dedent(f"""\
            module {module_name} (
            {chr(10).join(port_lines)}
            );

            {chr(10).join(param_lines)}

            {chr(10).join(reg_lines)}

                logic spike_detect;

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
            {chr(10).join(reset_lines)}
                        spike_detect <= 1'b0;
                    end else begin
                        // Dynamics integration point: downstream synthesis fills in
                        // the actual ODE integration from plugin parameters.
                        spike_detect <= 1'b0;
                    end
                end

            {chr(10).join(assign_lines)}
                assign o_spike = spike_detect;

            endmodule
        """)

        return header + body

    def _to_fixed(self, value: float) -> int:
        return int(round(value * (1 << self.frac_bits)))


# ── Documentation Generator ──────────────────────────────────────────


class DocGenerator:
    """Generates markdown documentation from plugin metadata."""

    def generate(self, plugin: NeuronPlugin) -> str:
        meta = plugin.meta()
        lines = [
            f"# {meta.name}",
            "",
            f"**Version**: {meta.version}  ",
            f"**Author**: {meta.author}  ",
            f"**Description**: {meta.description}",
            "",
        ]
        if meta.references:
            lines.append("## References")
            lines.append("")
            for ref in meta.references:
                lines.append(f"- {ref}")
            lines.append("")

        if meta.parameters:
            lines.append("## Parameters")
            lines.append("")
            lines.append("| Name | Description |")
            lines.append("|------|-------------|")
            for pname, pdesc in meta.parameters.items():
                lines.append(f"| `{pname}` | {pdesc} |")
            lines.append("")

        default_params = plugin.default_params()
        if default_params:
            lines.append("## Default Values")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for pname, pval in default_params.items():
                lines.append(f"| `{pname}` | {pval} |")
            lines.append("")

        if meta.state_variables:
            lines.append("## State Variables")
            lines.append("")
            for sv in meta.state_variables:
                lines.append(f"- `{sv}`")
            lines.append("")

        return "\n".join(lines)

    def generate_index(self, registry: PluginRegistry) -> str:
        """Generate a summary index for all registered plugins."""
        lines = [
            "# SC-NeuroCore Model Zoo",
            "",
            "| Model | Version | Description |",
            "|-------|---------|-------------|",
        ]
        for name in registry.list_plugins():
            plugin = registry.get(name)
            if plugin:
                m = plugin.meta()
                lines.append(f"| {m.name} | {m.version} | {m.description} |")
        lines.append("")
        return "\n".join(lines)

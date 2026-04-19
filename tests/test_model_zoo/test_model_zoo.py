# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model Zoo Tests

import numpy as np

from sc_neurocore.model_zoo.model_zoo import (
    AdExPlugin,
    DocGenerator,
    HodgkinHuxleyPlugin,
    IzhikevichPlugin,
    LIFPlugin,
    NeuronState,
    PluginRegistry,
    VerilogGenerator,
)


# ── NeuronState Tests ────────────────────────────────────────────────


class TestNeuronState:
    def test_get_set(self):
        s = NeuronState({"V": -65.0})
        assert s["V"] == -65.0
        s["V"] = -50.0
        assert s["V"] == -50.0

    def test_copy_independent(self):
        s = NeuronState({"V": -65.0})
        c = s.copy()
        c["V"] = 0.0
        assert s["V"] == -65.0

    def test_as_dict(self):
        s = NeuronState({"V": -65.0, "u": -14.0})
        d = s.as_dict()
        assert d == {"V": -65.0, "u": -14.0}


# ── Plugin Registry Tests ───────────────────────────────────────────


class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        reg.register(LIFPlugin())
        assert "LIF" in reg
        assert reg.get("LIF") is not None

    def test_list_plugins_sorted(self):
        reg = PluginRegistry.with_builtins()
        names = reg.list_plugins()
        assert names == sorted(names)
        assert len(names) == 4

    def test_builtins_all_present(self):
        reg = PluginRegistry.with_builtins()
        assert "LIF" in reg
        assert "Izhikevich" in reg
        assert "AdEx" in reg
        assert "Hodgkin-Huxley" in reg

    def test_get_missing_returns_none(self):
        reg = PluginRegistry()
        assert reg.get("nonexistent") is None

    def test_len(self):
        reg = PluginRegistry()
        assert len(reg) == 0
        reg.register(LIFPlugin())
        assert len(reg) == 1


# ── LIF Plugin Tests ─────────────────────────────────────────────────


class TestLIFPlugin:
    def test_meta_name(self):
        plugin = LIFPlugin()
        assert plugin.meta().name == "LIF"

    def test_default_state(self):
        plugin = LIFPlugin()
        state = plugin.default_state()
        assert "V" in state.as_dict()

    def test_subthreshold_no_spike(self):
        plugin = LIFPlugin()
        state = plugin.default_state()
        params = plugin.default_params()
        state = plugin.ode_dynamics(state, 0.0, params, 0.001)
        assert not plugin.threshold_check(state, params)

    def test_suprathreshold_spikes(self):
        plugin = LIFPlugin()
        params = plugin.default_params()
        current = np.ones(5000) * 2e-9
        _, spikes = plugin.simulate(current, dt=0.0001, params=params)
        assert len(spikes) > 0, "constant suprathreshold current should produce spikes"

    def test_reset_below_threshold(self):
        plugin = LIFPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": params["V_thresh"] + 0.01})
        reset_state = plugin.reset(state, params)
        assert reset_state["V"] == params["V_reset"]


# ── Izhikevich Plugin Tests ─────────────────────────────────────────


class TestIzhikevichPlugin:
    def test_meta_name(self):
        plugin = IzhikevichPlugin()
        assert plugin.meta().name == "Izhikevich"

    def test_state_variables(self):
        plugin = IzhikevichPlugin()
        state = plugin.default_state()
        assert "V" in state.as_dict()
        assert "u" in state.as_dict()

    def test_suprathreshold_spikes(self):
        plugin = IzhikevichPlugin()
        params = plugin.default_params()
        current = np.ones(10000) * 10.0
        _, spikes = plugin.simulate(current, dt=0.0001, params=params)
        assert len(spikes) > 0

    def test_reset_applies_d(self):
        plugin = IzhikevichPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": 35.0, "u": 0.0})
        reset_state = plugin.reset(state, params)
        assert reset_state["V"] == params["c"]
        assert reset_state["u"] == params["d"]


# ── AdEx Plugin Tests ────────────────────────────────────────────────


class TestAdExPlugin:
    def test_meta_name(self):
        plugin = AdExPlugin()
        assert plugin.meta().name == "AdEx"

    def test_state_has_adaptation(self):
        state = AdExPlugin().default_state()
        assert "w" in state.as_dict()

    def test_reset_increments_w(self):
        plugin = AdExPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": 25.0, "w": 1.0})
        reset = plugin.reset(state, params)
        assert reset["w"] == 1.0 + params["b"]


# ── Hodgkin-Huxley Plugin Tests ──────────────────────────────────────


class TestHodgkinHuxleyPlugin:
    def test_meta_name(self):
        plugin = HodgkinHuxleyPlugin()
        assert plugin.meta().name == "Hodgkin-Huxley"

    def test_four_state_variables(self):
        state = HodgkinHuxleyPlugin().default_state()
        d = state.as_dict()
        assert set(d.keys()) == {"V", "m", "h", "n"}

    def test_gating_variables_bounded(self):
        plugin = HodgkinHuxleyPlugin()
        params = plugin.default_params()
        state = plugin.default_state()
        for _ in range(100):
            state = plugin.ode_dynamics(state, 10.0, params, 0.0001)
        for gate in ("m", "h", "n"):
            assert 0.0 <= state[gate] <= 1.0, f"gate {gate} out of bounds"

    def test_reset_is_noop(self):
        plugin = HodgkinHuxleyPlugin()
        state = NeuronState({"V": 10.0, "m": 0.5, "h": 0.5, "n": 0.5})
        reset = plugin.reset(state, plugin.default_params())
        assert reset["V"] == 10.0


# ── Verilog Generator Tests ──────────────────────────────────────────


class TestVerilogGenerator:
    def test_generates_valid_module(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "module sc_neuron_lif" in sv
        assert "endmodule" in sv

    def test_contains_spdx_header(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "SPDX-License-Identifier" in sv
        assert "AGPL-3.0-or-later" in sv

    def test_ports_include_spike(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "o_spike" in sv

    def test_ports_include_state(self):
        gen = VerilogGenerator()
        sv = gen.generate(IzhikevichPlugin())
        assert "o_V" in sv
        assert "o_u" in sv

    def test_parameters_present(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "TAU_M" in sv or "V_REST" in sv

    def test_hh_generates_four_outputs(self):
        gen = VerilogGenerator()
        sv = gen.generate(HodgkinHuxleyPlugin())
        for var in ("o_V", "o_m", "o_h", "o_n"):
            assert var in sv

    def test_bit_width_configurable(self):
        gen = VerilogGenerator(bit_width=32, frac_bits=16)
        sv = gen.generate(LIFPlugin())
        assert "[31:0]" in sv

    def test_all_builtins_generate(self):
        gen = VerilogGenerator()
        for cls in (LIFPlugin, IzhikevichPlugin, AdExPlugin, HodgkinHuxleyPlugin):
            sv = gen.generate(cls())
            assert "module" in sv
            assert "endmodule" in sv


# ── Doc Generator Tests ──────────────────────────────────────────────


class TestDocGenerator:
    def test_generates_markdown(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert md.startswith("# LIF")
        assert "## Parameters" in md
        assert "## Default Values" in md

    def test_references_section(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "## References" in md
        assert "Lapicque" in md

    def test_state_variables_listed(self):
        doc = DocGenerator()
        md = doc.generate(IzhikevichPlugin())
        assert "## State Variables" in md
        assert "`V`" in md
        assert "`u`" in md

    def test_index_generation(self):
        reg = PluginRegistry.with_builtins()
        doc = DocGenerator()
        index = doc.generate_index(reg)
        assert "# SC-NeuroCore Model Zoo" in index
        assert "LIF" in index
        assert "Izhikevich" in index
        assert "AdEx" in index
        assert "Hodgkin-Huxley" in index

    def test_default_values_table(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "0.02" in md  # tau_m default

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python integration tests for the Rust engine

"""Integration tests exercising the Rust backend through the Python API.

Covers: NetworkRunner simulation, neuron step parity, IR compilation,
and spike data format. The Rust SIMD primitives are tested separately
in the 378 Rust-native tests (cargo test).
"""

import pytest

engine = pytest.importorskip("sc_neurocore_engine")


class TestNetworkRunner:
    def test_create_and_run(self) -> None:
        r = engine.NetworkRunner()
        idx = r.add_population("Izhikevich", 10)
        assert idx == 0
        results = r.run(100)
        assert "spike_data" in results
        assert "voltages" in results
        assert "spike_counts" in results

    def test_multiple_populations(self) -> None:
        r = engine.NetworkRunner()
        i0 = r.add_population("Izhikevich", 5)
        i1 = r.add_population("AdEx", 5)
        assert i0 == 0
        assert i1 == 1
        results = r.run(50)
        assert len(results["spike_data"]) == 2
        assert len(results["voltages"]) == 2

    def test_spike_data_u64_format(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Lapicque", 20)
        results = r.run(200)
        for packed in results["spike_data"][0]:
            nid = int(packed >> 32)
            t = int(packed & 0xFFFFFFFF)
            assert 0 <= nid < 20
            assert 0 <= t < 200

    def test_voltages_returned(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("HodgkinHuxley", 3)
        results = r.run(50)
        v = results["voltages"][0]
        assert len(v) == 3

    def test_projection_csr(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Izhikevich", 3)
        r.add_population("Izhikevich", 3)
        # All-to-all CSR: 3 source → 3 target, weight 0.5
        row_offsets = [0, 3, 6, 9]
        col_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        values = [0.5] * 9
        r.add_projection(0, 1, row_offsets, col_indices, values)
        results = r.run(100)
        assert len(results["spike_data"]) == 2

    def test_spike_counts(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Izhikevich", 10)
        results = r.run(100)
        counts = results["spike_counts"]
        assert len(counts) == 1
        assert counts[0] >= 0


class TestRustNeurons:
    @pytest.mark.parametrize(
        ("model", "current"),
        [
            ("Izhikevich", 15.0),
            ("HodgkinHuxleyNeuron", 15.0),
            ("AdExNeuron", 200.0),
            ("LapicqueNeuron", 15.0),
        ],
    )
    def test_neuron_produces_spikes(self, model: str, current: float) -> None:
        cls = getattr(engine, model)
        neuron = cls()
        spikes = sum(neuron.step(current) for _ in range(500))
        assert spikes > 0

    def test_izhikevich_deterministic(self) -> None:
        a = engine.Izhikevich()
        b = engine.Izhikevich()
        sa = [a.step(10.0) for _ in range(100)]
        sb = [b.step(10.0) for _ in range(100)]
        assert sa == sb

    def test_izhikevich_reset(self) -> None:
        n = engine.Izhikevich()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        fresh = engine.Izhikevich()
        assert n.step(0.0) == fresh.step(0.0)

    def test_arcane_neuron_exists(self) -> None:
        n = engine.ArcaneNeuron()
        spike = n.step(5.0)
        assert spike in (0, 1)


class TestIRCompiler:
    def test_build_verify_emit(self) -> None:
        b = engine.ScGraphBuilder("test_lif")
        i_in = b.input("current", "bool")
        leak = b.constant_i64(200, "i16")
        gain = b.constant_i64(256, "i16")
        noise = b.constant_i64(0, "i16")
        v_lif = b.lif_step(i_in, leak, gain, noise)
        b.output("spike", v_lif)
        graph = b.build()
        assert graph.verify() is None
        sv = graph.emit_sv()
        assert "module" in sv

    def test_ir_print(self) -> None:
        b = engine.ScGraphBuilder("print_test")
        v = b.input("x", "bool")
        b.output("y", v)
        graph = b.build()
        text = graph.to_text()
        assert "print_test" in text

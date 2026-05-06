# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore


class TestTMRWrapper:
    """Triple Modular Redundancy wrapper generation."""

    def test_majority_voter_structure(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_tmr" in v
        assert "endmodule" in v
        assert "inst_a" in v
        assert "inst_b" in v
        assert "inst_c" in v
        assert "seu_detected" in v

    def test_median_voter(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_hh", data_width=32, voter="median")
        assert "Median" in v
        assert "sc_hh_tmr" in v

    def test_multi_state_var(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_izh", state_vars=["v", "u"])
        assert "v_voted" in v
        assert "u_voted" in v
        assert "v_a" in v
        assert "u_c" in v

    def test_seu_detection_wires(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif")
        assert "(v_a != v_b)" in v
        assert "(spike_a != spike_b)" in v

    def test_tmr_references_inner_module(self):
        from sc_neurocore.compiler.intelligence import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_custom_neuron")
        assert "sc_custom_neuron inst_a" in v
        assert "sc_custom_neuron inst_b" in v
        assert "sc_custom_neuron inst_c" in v


class TestPIMLayout:
    """Processing-in-Memory data layout planning."""

    def test_basic_layout(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000, 10000)
        assert layout.bank_count >= 1
        assert layout.neurons_per_bank >= 1
        assert layout.weights_per_bank >= 1
        assert 0 < layout.bank_utilisation <= 1.0
        assert layout.parallel_factor >= 1

    def test_layout_map_regions(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000, 50000, num_banks=16)
        assert "neuron_state" in layout.layout_map
        assert "synaptic_weights" in layout.layout_map

    def test_large_network_uses_more_banks(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        small = plan_pim_layout(100, 1000, num_banks=16)
        large = plan_pim_layout(100000, 10000000, num_banks=16)
        assert large.bank_count >= small.bank_count

    def test_respects_bank_limit(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000000, 100000000, num_banks=8)
        assert layout.bank_count <= 8

    def test_custom_bank_size(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(100, 1000, bank_size_kb=32)
        assert layout.bank_count >= 1


class TestPowerDomainWrapper:
    """Clock/power gating wrapper for edge deployment."""

    def test_basic_structure(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_pg" in v
        assert "endmodule" in v
        assert "power_down" in v
        assert "power_state" in v

    def test_icg_cell(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "gated_clk" in v
        assert "clk_enable" in v

    def test_wakeup_counter(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", wakeup_cycles=8)
        assert "wakeup_cnt" in v
        assert "active" in v

    def test_state_retention(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper(
            "sc_izh",
            state_vars=["v", "u"],
        )
        assert "v_out" in v
        assert "u_out" in v
        assert "retain" in v.lower()

    def test_always_on_domain(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "Always-on" in v
        assert "spike_out" in v


class TestUCIePartitioning:
    """Chiplet die-to-die neuron array partitioning."""

    def test_basic_partition(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(1000, 0.1, tile_count=4)
        assert p.tile_count == 4
        assert p.neurons_per_tile == 250
        assert p.die_to_die_bandwidth_gbps >= 0

    def test_partition_map_covers_all_neurons(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=4)
        all_neurons = []
        for ids in p.partition_map.values():
            all_neurons.extend(ids)
        assert len(set(all_neurons)) == 100

    def test_more_tiles_more_inter_traffic(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p2 = advise_ucie_partition(1000, 0.1, tile_count=2)
        p8 = advise_ucie_partition(1000, 0.1, tile_count=8)
        # More tiles → more inter-tile fraction → more bandwidth
        assert p8.die_to_die_bandwidth_gbps >= p2.die_to_die_bandwidth_gbps

    def test_latency_scales_with_tiles(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p2 = advise_ucie_partition(100, 0.1, tile_count=2)
        p8 = advise_ucie_partition(100, 0.1, tile_count=8)
        assert p8.latency_penalty_ns > p2.latency_penalty_ns

    def test_single_tile_no_overhead(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=1)
        assert p.inter_tile_spikes == 0
        assert p.latency_penalty_ns == 0


class TestCXLCoherence:
    """CXL.mem Type-3 device mapping."""

    def test_basic_mapping(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(10000, 1000000)
        assert m.device_count >= 1
        assert len(m.state_device_ids) >= 1
        assert len(m.weight_device_ids) >= 1
        assert m.total_capacity_gb > 0

    def test_streaming_uses_cxl_mem(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="streaming")
        assert m.coherence_protocol == "CXL.mem"

    def test_random_uses_cxl_cache(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="random")
        assert m.coherence_protocol == "CXL.cache"

    def test_respects_device_limit(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        m = advise_cxl_mapping(
            1000000000,
            10000000000,
            max_devices=4,
        )
        assert m.device_count <= 4

    def test_random_needs_more_bandwidth(self):
        from sc_neurocore.compiler.intelligence import advise_cxl_mapping

        s = advise_cxl_mapping(10000, 1000000, access_pattern="streaming")
        r = advise_cxl_mapping(10000, 1000000, access_pattern="random")
        assert r.host_bandwidth_gbps > s.host_bandwidth_gbps


class TestPipelineWrapper:
    """Pipeline register insertion for high-frequency targets."""

    def test_basic_pipeline(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b + c"},
            data_width=16,
        )
        assert "module sc_lif_pipe" in v
        assert "endmodule" in v
        assert "valid_in" in v
        assert "valid_out" in v
        assert "latency" in v

    def test_pipeline_stages_in_output(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_hh",
            {"v": "a * b * c"},
            stages=3,
        )
        assert "I_pipe_0" in v
        assert "I_pipe_1" in v
        assert "I_pipe_2" in v
        assert "valid_pipe" in v

    def test_auto_stages_from_target(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b * c * d * e"},
            target="artix7",
        )
        assert "module sc_lif_pipe" in v
        assert "pipeline" in v.lower()

    def test_output_register(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_lif", {"v": "a * b"})
        assert "v_out" in v
        assert "spike_out" in v
        assert "v_reg" in v

    def test_inner_module_instantiation(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_custom", {"v": "a + b"})
        assert "sc_custom core" in v


class TestCrossFeatureIntegration:
    """End-to-end tests chaining multiple features together."""

    def test_tmr_plus_checksum(self):
        """TMR wrapper + model checksum embedding."""
        from sc_neurocore.compiler.intelligence import (
            generate_tmr_wrapper,
            embed_model_checksum,
        )

        tmr = generate_tmr_wrapper("sc_lif", data_width=16)
        result = embed_model_checksum(
            tmr,
            equations={"v": "a + b"},
            params={"tmr": True},
        )
        assert "sc_lif_tmr" in result
        assert "MODEL_HASH" in result

    def test_pipeline_plus_power_domain(self):
        """Pipeline wrapper output feeds power-domain wrapper input."""
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
            generate_power_domain_wrapper,
        )

        pipe = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b"},
            stages=2,
        )
        pg = generate_power_domain_wrapper("sc_lif_pipe")
        assert "sc_lif_pipe" in pipe
        assert "sc_lif_pipe_pg" in pg

    def test_mzi_then_noise(self):
        """Encode weights for photonic, then inject noise for robustness."""
        from sc_neurocore.compiler.intelligence import (
            encode_mzi_weights,
            inject_weight_noise,
        )

        weights = [[1.0, -0.5], [0.3, 0.8]]
        enc = encode_mzi_weights(weights)
        noisy = inject_weight_noise(weights, seed=42)
        enc_noisy = encode_mzi_weights(noisy)
        # Noisy encoding should differ from clean
        assert enc.phases_theta != enc_noisy.phases_theta

    def test_quant_sweep_then_compare(self):
        """Sweep quantisation, then compare top 2 widths on 2 targets."""
        from sc_neurocore.compiler.intelligence import (
            auto_quantisation_sweep,
            compare_targets,
        )

        sweep = auto_quantisation_sweep({"v": "a * b + c"}, widths=[8, 16])
        assert len(sweep) == 2
        cmp = compare_targets({"v": "a * b + c"}, ["artix7", "loihi2"])
        assert len(cmp) == 2
        # Both should have valid data
        assert sweep[0].data_width < sweep[1].data_width
        assert cmp[0].target != cmp[1].target

    def test_full_compilation_pipeline(self):
        """Full pipeline: compile → summary → checksum → encrypt."""
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
            embed_model_checksum,
            generate_bitstream_encryption,
        )

        eqs = {"v": "0.04 * v * v + 5 * v + 140 - u + I", "u": "a * (b * v - u)"}
        # Step 1: Summary
        summary = generate_compilation_summary("sc_izh", eqs, "artix7")
        assert "sc_izh" in summary
        # Step 2: Checksum
        verilog = "module sc_izh(...);\nendmodule"
        hashed = embed_model_checksum(verilog, equations=eqs)
        assert "MODEL_HASH" in hashed
        # Step 3: Encryption
        enc = generate_bitstream_encryption("sc_izh")
        assert "ENCRYPT" in enc


class TestUCIeMapper:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import map_ucie_protocol

        r = map_ucie_protocol({"core_a": 64, "core_b": 128})
        assert r.lanes["core_a"] >= 1
        assert r.lanes["core_b"] >= 1
        assert r.total_bandwidth_gbps > 0
        assert "UCIe" in r.protocol_version


class TestBRAMArray:
    """Tests for BRAM-backed neuron array generation."""

    def test_basic_array(self):
        """Default array generates valid Verilog."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array()
        assert "module sc_neuron_array" in v
        assert "state_bram" in v
        assert "ram_style" in v
        assert "endmodule" in v

    def test_custom_count(self):
        """Custom neuron count."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array(neuron_count=256)
        assert "[0:255]" in v

    def test_custom_module_name(self):
        """Custom module name."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array(module_name="my_array")
        assert "module my_array" in v

    def test_spike_output(self):
        """Array has spike output ports."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array()
        assert "spike_out" in v
        assert "spike_neuron_id" in v
        assert "tick_done" in v


class TestMemoryMap:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_memory_map

        m = generate_memory_map("sc_lif", {"v": "a", "u": "b"})
        assert m.total_bytes > 0
        assert "addr_dec" in m.decoder_verilog
        assert len(m.entries) > 0

    def test_base_address(self):
        from sc_neurocore.compiler.intelligence import generate_memory_map

        m = generate_memory_map("sc_lif", {"v": "a"}, base_address=0x2000)
        assert m.base_address == 0x2000


class TestFloorplanner:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        r = plan_multi_die_floorplan(
            {"cortex_a": 500, "cortex_b": 300, "cortex_c": 400},
            die_capacity=1000,
        )
        assert len(r.die_assignment) == 3
        assert r.total_dies >= 1

    def test_overflow(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        r = plan_multi_die_floorplan(
            {"big": 900, "huge": 800},
            die_capacity=1000,
            num_dies=2,
        )
        assert r.total_dies == 2

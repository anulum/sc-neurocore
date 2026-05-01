# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851

"""Wave 5 tests — universal hardware coverage & strategic features.

Tests for:
- 36 new hardware profiles (photonic, chiplet, PIM/CXL, neuromorphic,
  sovereign/defence, automotive/edge)
- User-defined TOML profile loader
- SEU/TMR wrapper generator
- Model checksum embedding
- Auto-quantisation sweep
"""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. New Platform Profiles — Existence & Q-Format Validation
# ═══════════════════════════════════════════════════════════════════════


class TestPhotonicProfiles:
    """Photonic / optical compute platform profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "lightmatter_passage",
            "lightelligence_pace",
            "xanadu_x8",
            "ipronics_smartlight",
            "luminous_computing",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.platform_class == "photonic"
        assert p.data_width > 0
        assert p.fraction < p.data_width

    def test_mzi_dsp_block(self):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("lightmatter_passage")
        assert p.dsp_block == "MZI"


class TestChipletProfiles:
    """Chiplet / UCIe / heterogeneous integration profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "tenstorrent_blackhole",
            "cerebras_wse3",
            "intel_ponte_vecchio",
            "amd_mi300x",
            "ucie_generic",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_wse3_frequency(self):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("cerebras_wse3")
        assert p.max_freq_mhz == 1000


class TestPIMCXLProfiles:
    """Processing-in-memory and CXL memory profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "upmem_pim",
            "samsung_hbm_pim",
            "sk_hynix_aim",
            "cxl_type3",
            "axdimm",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.platform_class == "in_memory"


class TestNextGenNeuromorphicProfiles:
    """Next-generation neuromorphic platform profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "akida2",
            "spinnaker2",
            "dynapse2",
            "rain_neuromorphic",
            "brainscales2",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.platform_class == "neuromorphic"

    def test_brainscales2_wrap(self):
        """BrainScaleS-2 uses wrap overflow (analog domain)."""
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("brainscales2")
        assert p.overflow == "wrap"


class TestSovereignDefenceProfiles:
    """Sovereign / defence / aerospace profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "bae_rad750",
            "cobham_ut700",
            "mpfs250t_rt",
            "versal_xqrvc1902",
            "trenz_zynq_space",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.platform_class == "fpga"

    def test_rad750_no_dsp(self):
        """RAD750 has no dedicated DSP blocks."""
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("bae_rad750")
        assert p.dsp_block == ""
        assert p.dsp_mult_a == 0


class TestAutomotiveEdgeProfiles:
    """Automotive / edge AI SoC profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "mythic_m1076",
            "mobileye_eyeq6",
            "horizon_j6",
            "ambarella_cv72s",
            "hailo15",
            "syntiant_ndp120",
        ],
    )
    def test_profile_exists(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.data_width > 0


class TestTotalProfileCount:
    """Verify total platform coverage."""

    def test_at_least_113_profiles(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names

        names = list_profile_names()
        assert len(names) >= 113, f"Only {len(names)} profiles found"

    def test_10_platform_classes(self):
        from sc_neurocore.compiler.hardware_profiles import list_profiles

        classes = {p.platform_class for p in list_profiles()}
        assert len(classes) >= 9

    def test_filter_by_photonic(self):
        from sc_neurocore.compiler.hardware_profiles import list_profiles

        photonic = list_profiles(platform_class="photonic")
        assert len(photonic) >= 5

    def test_filter_by_in_memory(self):
        from sc_neurocore.compiler.hardware_profiles import list_profiles

        pim = list_profiles(platform_class="in_memory")
        assert len(pim) >= 5


# ═══════════════════════════════════════════════════════════════════════
# 2. User-Defined TOML Profile Loader
# ═══════════════════════════════════════════════════════════════════════


class TestTOMLProfileLoader:
    """TOML-based custom profile registration."""

    def test_load_valid_toml(self, tmp_path):
        from sc_neurocore.compiler.hardware_profiles import (
            load_toml_profile,
            get_profile,
        )

        toml = tmp_path / "test_chip.toml"
        toml.write_text("""\
[profile]
name = "test_my_chip_v1"
vendor = "TestCorp"
family = "TestNet-1"
platform_class = "accelerator"
data_width = 16
fraction = 8
overflow = "saturate"
rounding = "nearest"
max_freq_mhz = 500
dsp_block = "MAC"
dsp_mult_a = 16
dsp_mult_b = 16
notes = "Custom test chip."
""")
        p = load_toml_profile(str(toml))
        assert p.name == "test_my_chip_v1"
        assert p.vendor == "TestCorp"
        assert p.data_width == 16
        assert p.max_freq_mhz == 500

        # Should be retrievable
        p2 = get_profile("test_my_chip_v1")
        assert p2.vendor == "TestCorp"

    def test_load_missing_file(self):
        from sc_neurocore.compiler.hardware_profiles import load_toml_profile

        with pytest.raises(FileNotFoundError):
            load_toml_profile("/nonexistent/path/chip.toml")

    def test_load_missing_fields(self, tmp_path):
        from sc_neurocore.compiler.hardware_profiles import load_toml_profile

        toml = tmp_path / "incomplete.toml"
        toml.write_text("""\
[profile]
name = "incomplete_chip"
vendor = "TestCorp"
""")
        with pytest.raises(ValueError, match="Missing required fields"):
            load_toml_profile(str(toml))

    def test_load_toml_dir(self, tmp_path):
        from sc_neurocore.compiler.hardware_profiles import load_toml_profiles_dir

        for i in range(3):
            t = tmp_path / f"chip_{i}.toml"
            t.write_text(f"""\
[profile]
name = "test_dir_chip_{i}"
vendor = "DirCorp"
family = "Dir-{i}"
platform_class = "accelerator"
data_width = 16
fraction = 8
overflow = "saturate"
rounding = "nearest"
""")
        loaded = load_toml_profiles_dir(str(tmp_path))
        assert len(loaded) == 3

    def test_load_empty_dir(self, tmp_path):
        from sc_neurocore.compiler.hardware_profiles import load_toml_profiles_dir

        loaded = load_toml_profiles_dir(str(tmp_path))
        assert len(loaded) == 0

    def test_minimal_toml(self, tmp_path):
        """Minimal TOML without optional fields."""
        from sc_neurocore.compiler.hardware_profiles import load_toml_profile

        toml = tmp_path / "minimal.toml"
        toml.write_text("""\
[profile]
name = "test_minimal_chip"
vendor = "MinCorp"
family = "Min-1"
platform_class = "emerging"
data_width = 8
fraction = 4
overflow = "wrap"
rounding = "truncate"
""")
        p = load_toml_profile(str(toml))
        assert p.dsp_block == ""
        assert p.max_freq_mhz is None
        assert p.notes == "User-defined profile."


# ═══════════════════════════════════════════════════════════════════════
# 3. SEU / TMR Wrapper Generator
# ═══════════════════════════════════════════════════════════════════════


class TestTMRWrapper:
    """Triple Modular Redundancy wrapper generation."""

    def test_majority_voter_structure(self):
        from sc_neurocore.compiler.advanced_features import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_tmr" in v
        assert "endmodule" in v
        assert "inst_a" in v
        assert "inst_b" in v
        assert "inst_c" in v
        assert "seu_detected" in v

    def test_median_voter(self):
        from sc_neurocore.compiler.advanced_features import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_hh", data_width=32, voter="median")
        assert "Median" in v
        assert "sc_hh_tmr" in v

    def test_multi_state_var(self):
        from sc_neurocore.compiler.advanced_features import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_izh", state_vars=["v", "u"])
        assert "v_voted" in v
        assert "u_voted" in v
        assert "v_a" in v
        assert "u_c" in v

    def test_seu_detection_wires(self):
        from sc_neurocore.compiler.advanced_features import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_lif")
        assert "(v_a != v_b)" in v
        assert "(spike_a != spike_b)" in v

    def test_tmr_references_inner_module(self):
        from sc_neurocore.compiler.advanced_features import generate_tmr_wrapper

        v = generate_tmr_wrapper("sc_custom_neuron")
        assert "sc_custom_neuron inst_a" in v
        assert "sc_custom_neuron inst_b" in v
        assert "sc_custom_neuron inst_c" in v


# ═══════════════════════════════════════════════════════════════════════
# 4. Model Checksum / Hash Embedding
# ═══════════════════════════════════════════════════════════════════════


class TestModelChecksum:
    """SHA-256 model checksum embedding."""

    def test_checksum_embedded(self):
        from sc_neurocore.compiler.advanced_features import embed_model_checksum

        verilog = "// Test module\nmodule sc_lif (...);\nendmodule"
        result = embed_model_checksum(
            verilog,
            equations={"v": "a + b"},
            params={"data_width": 16},
        )
        assert "SHA-256:" in result
        assert "MODEL_HASH" in result
        assert "256'h" in result

    def test_checksum_deterministic(self):
        from sc_neurocore.compiler.advanced_features import embed_model_checksum

        v1 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        v2 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        # Same inputs → same hash
        import re

        h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1).group(1)
        h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2).group(1)
        assert h1 == h2

    def test_different_equations_different_hash(self):
        from sc_neurocore.compiler.advanced_features import embed_model_checksum
        import re

        v1 = embed_model_checksum("module x; endmodule", equations={"v": "a+b"})
        v2 = embed_model_checksum("module x; endmodule", equations={"v": "a*b"})
        h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1).group(1)
        h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2).group(1)
        assert h1 != h2

    def test_no_equations_still_works(self):
        from sc_neurocore.compiler.advanced_features import embed_model_checksum

        result = embed_model_checksum("module y; endmodule")
        assert "MODEL_HASH" in result


# ═══════════════════════════════════════════════════════════════════════
# 5. Auto-Quantisation Sweep
# ═══════════════════════════════════════════════════════════════════════


class TestQuantisationSweep:
    """Quantisation design-space exploration."""

    def test_default_sweep(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b + c"})
        assert len(results) == 7  # [4, 8, 12, 16, 20, 24, 32]

    def test_widths_sorted(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b"})
        widths = [r.data_width for r in results]
        assert widths == sorted(widths)

    def test_luts_grow_with_width(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b + c"})
        luts = [r.estimated_luts for r in results]
        assert luts == sorted(luts)  # Monotonically increasing

    def test_precision_improves_with_width(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b"})
        steps = [r.min_step for r in results]
        assert steps == sorted(steps, reverse=True)  # Smaller step = better

    def test_custom_widths(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {"v": "a + b"},
            widths=[8, 16, 32],
        )
        assert len(results) == 3

    def test_format_report(self):
        from sc_neurocore.compiler.advanced_features import (
            auto_quantisation_sweep,
            format_quantisation_report,
        )

        results = auto_quantisation_sweep({"v": "a * b + c"})
        report = format_quantisation_report(results)
        assert "Q-format" in report
        assert "LUTs" in report
        assert "LSB Step" in report

    def test_target_affects_dsps(self):
        """Targets with DSP blocks should show DSP usage."""
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        r_artix = auto_quantisation_sweep(
            {"v": "a * b"},
            target="artix7",
        )
        r_ice40 = auto_quantisation_sweep(
            {"v": "a * b"},
            target="bae_rad750",
        )
        # Artix has DSP48E1, RAD750 doesn't
        assert all(r.estimated_dsps > 0 for r in r_artix)
        assert all(r.estimated_dsps == 0 for r in r_ice40)

    def test_izh_multi_equation(self):
        from sc_neurocore.compiler.advanced_features import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {
                "v": "0.04 * v * v + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            }
        )
        # More equations → more FFs
        for r in results:
            assert r.estimated_ffs == 2 * r.data_width  # 2 state vars


# ═══════════════════════════════════════════════════════════════════════
# 6. MZI / Optical Weight Encoding
# ═══════════════════════════════════════════════════════════════════════


class TestMZIWeightEncoding:
    """Photonic MZI phase-shift weight encoding."""

    def test_encode_identity_matrix(self):
        from sc_neurocore.compiler.advanced_features import encode_mzi_weights

        weights = [[1.0, 0.0], [0.0, 1.0]]
        enc = encode_mzi_weights(weights)
        assert enc.mesh_size == 2
        assert len(enc.phases_theta) == 2
        assert len(enc.phases_theta[0]) == 2

    def test_negative_weights_use_pi_shift(self):
        import math
        from sc_neurocore.compiler.advanced_features import encode_mzi_weights

        weights = [[-1.0, 0.5]]
        enc = encode_mzi_weights(weights)
        # Negative weight → φ = π
        assert enc.phases_phi[0][0] == round(math.pi, 6)
        # Positive weight → φ = 0
        assert enc.phases_phi[0][1] == 0.0

    def test_theta_range(self):
        import math
        from sc_neurocore.compiler.advanced_features import encode_mzi_weights

        weights = [[0.0, 0.5, 1.0]]
        enc = encode_mzi_weights(weights)
        for theta in enc.phases_theta[0]:
            assert 0 <= theta <= math.pi + 0.001

    def test_loss_reduces_transmission(self):
        from sc_neurocore.compiler.advanced_features import encode_mzi_weights

        w = [[1.0]]
        enc_low = encode_mzi_weights(w, loss_db_per_mzi=0.0)
        enc_high = encode_mzi_weights(w, loss_db_per_mzi=3.0)
        assert enc_low.transmission[0][0] > enc_high.transmission[0][0]

    def test_json_config_output(self):
        import json
        from sc_neurocore.compiler.advanced_features import (
            encode_mzi_weights,
            generate_mzi_config,
        )

        enc = encode_mzi_weights([[1.0, -0.5], [0.3, 0.8]])
        cfg = generate_mzi_config(enc, output_format="json")
        data = json.loads(cfg)
        assert "mesh_size" in data
        assert "phases_theta" in data

    def test_csv_config_output(self):
        from sc_neurocore.compiler.advanced_features import (
            encode_mzi_weights,
            generate_mzi_config,
        )

        enc = encode_mzi_weights([[1.0, -0.5]])
        cfg = generate_mzi_config(enc, output_format="csv")
        assert "row,col,theta,phi,transmission" in cfg
        lines = cfg.strip().split("\n")
        assert len(lines) == 3  # header + 2 entries

    def test_zero_matrix_encoding(self):
        from sc_neurocore.compiler.advanced_features import encode_mzi_weights

        enc = encode_mzi_weights([[0.0, 0.0], [0.0, 0.0]])
        # All theta should be 0 (no transmission)
        for row in enc.phases_theta:
            for t in row:
                assert t == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 7. PIM Data Layout Planner
# ═══════════════════════════════════════════════════════════════════════


class TestPIMLayout:
    """Processing-in-Memory data layout planning."""

    def test_basic_layout(self):
        from sc_neurocore.compiler.advanced_features import plan_pim_layout

        layout = plan_pim_layout(1000, 10000)
        assert layout.bank_count >= 1
        assert layout.neurons_per_bank >= 1
        assert layout.weights_per_bank >= 1
        assert 0 < layout.bank_utilisation <= 1.0
        assert layout.parallel_factor >= 1

    def test_layout_map_regions(self):
        from sc_neurocore.compiler.advanced_features import plan_pim_layout

        layout = plan_pim_layout(1000, 50000, num_banks=16)
        assert "neuron_state" in layout.layout_map
        assert "synaptic_weights" in layout.layout_map

    def test_large_network_uses_more_banks(self):
        from sc_neurocore.compiler.advanced_features import plan_pim_layout

        small = plan_pim_layout(100, 1000, num_banks=16)
        large = plan_pim_layout(100000, 10000000, num_banks=16)
        assert large.bank_count >= small.bank_count

    def test_respects_bank_limit(self):
        from sc_neurocore.compiler.advanced_features import plan_pim_layout

        layout = plan_pim_layout(1000000, 100000000, num_banks=8)
        assert layout.bank_count <= 8

    def test_custom_bank_size(self):
        from sc_neurocore.compiler.advanced_features import plan_pim_layout

        layout = plan_pim_layout(100, 1000, bank_size_kb=32)
        assert layout.bank_count >= 1


# ═══════════════════════════════════════════════════════════════════════
# 8. Power Domain / Clock Gating Wrapper
# ═══════════════════════════════════════════════════════════════════════


class TestPowerDomainWrapper:
    """Clock/power gating wrapper for edge deployment."""

    def test_basic_structure(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_pg" in v
        assert "endmodule" in v
        assert "power_down" in v
        assert "power_state" in v

    def test_icg_cell(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "gated_clk" in v
        assert "clk_enable" in v

    def test_wakeup_counter(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", wakeup_cycles=8)
        assert "wakeup_cnt" in v
        assert "active" in v

    def test_state_retention(self):
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "Always-on" in v
        assert "spike_out" in v


# ═══════════════════════════════════════════════════════════════════════
# 9. HLS-C Export
# ═══════════════════════════════════════════════════════════════════════


class TestHLSCppExport:
    """Vitis/Catapult HLS C++ translation."""

    def test_vitis_export(self):
        from sc_neurocore.compiler.advanced_features import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_lif",
            {"v": "v + I_t - v * leak"},
            data_width=16,
            fraction=8,
        )
        assert "ap_fixed<16,8>" in cpp
        assert "#pragma HLS PIPELINE" in cpp
        assert "void sc_lif(" in cpp
        assert "V_THRESH" in cpp

    def test_catapult_export(self):
        from sc_neurocore.compiler.advanced_features import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_hh",
            {"v": "a + b", "n": "c * d"},
            hls_tool="catapult",
        )
        assert "Catapult" in cpp
        assert "v_next" in cpp
        assert "n_next" in cpp

    def test_include_guard(self):
        from sc_neurocore.compiler.advanced_features import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "a + b"})
        assert "#ifndef SC_LIF_HLS_H" in cpp
        assert "#endif" in cpp

    def test_state_struct(self):
        from sc_neurocore.compiler.advanced_features import generate_hls_cpp

        cpp = generate_hls_cpp("sc_izh", {"v": "a", "u": "b"})
        assert "struct sc_izh_state" in cpp
        assert "fp_t v;" in cpp
        assert "fp_t u;" in cpp

    def test_spike_detection(self):
        from sc_neurocore.compiler.advanced_features import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "v + I_t"})
        assert "spike_out" in cpp
        assert "V_THRESH" in cpp


# ═══════════════════════════════════════════════════════════════════════
# 10. Bitstream Encryption Wrapper
# ═══════════════════════════════════════════════════════════════════════


class TestBitstreamEncryption:
    """AES-256 bitstream encryption for secure boot."""

    def test_xilinx_encryption(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_lif", vendor="xilinx")
        assert "BITSTREAM.ENCRYPTION.ENCRYPT YES" in tcl
        assert "sc_lif.nky" in tcl
        assert "SECURITY_LEVEL" in tcl

    def test_intel_encryption(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_hh", vendor="intel")
        assert "ENCRYPTION_KEY_SOURCE" in tcl
        assert "ENABLE_CONFIGURATION_BITSTREAM_ENCRYPTION ON" in tcl
        assert "ANTI_TAMPER" in tcl

    def test_key_source_efuse(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="efuse",
        )
        assert "EFUSE" in tcl

    def test_key_source_bbram(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="bbram",
        )
        assert "BBRAM" in tcl

    def test_module_name_in_output(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("my_neuron_design")
        assert "my_neuron_design" in tcl


# ═══════════════════════════════════════════════════════════════════════
# 11. UCIe Partitioning Advisor
# ═══════════════════════════════════════════════════════════════════════


class TestUCIePartitioning:
    """Chiplet die-to-die neuron array partitioning."""

    def test_basic_partition(self):
        from sc_neurocore.compiler.advanced_features import advise_ucie_partition

        p = advise_ucie_partition(1000, 0.1, tile_count=4)
        assert p.tile_count == 4
        assert p.neurons_per_tile == 250
        assert p.die_to_die_bandwidth_gbps >= 0

    def test_partition_map_covers_all_neurons(self):
        from sc_neurocore.compiler.advanced_features import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=4)
        all_neurons = []
        for ids in p.partition_map.values():
            all_neurons.extend(ids)
        assert len(set(all_neurons)) == 100

    def test_more_tiles_more_inter_traffic(self):
        from sc_neurocore.compiler.advanced_features import advise_ucie_partition

        p2 = advise_ucie_partition(1000, 0.1, tile_count=2)
        p8 = advise_ucie_partition(1000, 0.1, tile_count=8)
        # More tiles → more inter-tile fraction → more bandwidth
        assert p8.die_to_die_bandwidth_gbps >= p2.die_to_die_bandwidth_gbps

    def test_latency_scales_with_tiles(self):
        from sc_neurocore.compiler.advanced_features import advise_ucie_partition

        p2 = advise_ucie_partition(100, 0.1, tile_count=2)
        p8 = advise_ucie_partition(100, 0.1, tile_count=8)
        assert p8.latency_penalty_ns > p2.latency_penalty_ns

    def test_single_tile_no_overhead(self):
        from sc_neurocore.compiler.advanced_features import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=1)
        assert p.inter_tile_spikes == 0
        assert p.latency_penalty_ns == 0


# ═══════════════════════════════════════════════════════════════════════
# 12. CXL Coherence Advisor
# ═══════════════════════════════════════════════════════════════════════


class TestCXLCoherence:
    """CXL.mem Type-3 device mapping."""

    def test_basic_mapping(self):
        from sc_neurocore.compiler.advanced_features import advise_cxl_mapping

        m = advise_cxl_mapping(10000, 1000000)
        assert m.device_count >= 1
        assert len(m.state_device_ids) >= 1
        assert len(m.weight_device_ids) >= 1
        assert m.total_capacity_gb > 0

    def test_streaming_uses_cxl_mem(self):
        from sc_neurocore.compiler.advanced_features import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="streaming")
        assert m.coherence_protocol == "CXL.mem"

    def test_random_uses_cxl_cache(self):
        from sc_neurocore.compiler.advanced_features import advise_cxl_mapping

        m = advise_cxl_mapping(1000, 10000, access_pattern="random")
        assert m.coherence_protocol == "CXL.cache"

    def test_respects_device_limit(self):
        from sc_neurocore.compiler.advanced_features import advise_cxl_mapping

        m = advise_cxl_mapping(
            1000000000,
            10000000000,
            max_devices=4,
        )
        assert m.device_count <= 4

    def test_random_needs_more_bandwidth(self):
        from sc_neurocore.compiler.advanced_features import advise_cxl_mapping

        s = advise_cxl_mapping(10000, 1000000, access_pattern="streaming")
        r = advise_cxl_mapping(10000, 1000000, access_pattern="random")
        assert r.host_bandwidth_gbps > s.host_bandwidth_gbps


# ═══════════════════════════════════════════════════════════════════════
# 13. On-Chip Learning Export
# ═══════════════════════════════════════════════════════════════════════


class TestOnChipLearning:
    """STDP / reward-modulated learning parameter export."""

    def test_default_stdp_params(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_learning_params,
        )

        p = generate_learning_params()
        assert p.learning_rule == "stdp"
        assert p.tau_plus_ms == 20.0
        assert p.a_plus == 0.01
        assert p.target_platform == "akida2"

    def test_rstdp_params(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_learning_params,
        )

        p = generate_learning_params(
            learning_rule="rstdp",
            reward_tau_ms=500.0,
        )
        assert p.learning_rule == "rstdp"
        assert p.reward_tau_ms == 500.0

    def test_json_export(self):
        import json
        from sc_neurocore.compiler.advanced_features import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params()
        cfg = export_learning_config(p, output_format="json")
        data = json.loads(cfg)
        assert data["learning_rule"] == "stdp"
        assert "time_constants" in data
        assert "weight_bounds" in data

    def test_yaml_export(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params(target="brainscales2")
        cfg = export_learning_config(p, output_format="yaml")
        assert "learning_rule: stdp" in cfg
        assert "brainscales2" in cfg
        assert "tau_plus_ms:" in cfg

    def test_custom_weight_bounds(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_learning_params,
        )

        p = generate_learning_params(w_max=2.0, w_min=-1.0)
        assert p.w_max == 2.0
        assert p.w_min == -1.0


# ═══════════════════════════════════════════════════════════════════════
# 14. Stochastic Weight Noise Injection
# ═══════════════════════════════════════════════════════════════════════


class TestWeightNoise:
    """Device-variation noise injection for analog robustness."""

    def test_gaussian_noise_changes_weights(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, -1.0], [0.5, 0.0]]
        noisy = inject_weight_noise(w, seed=42)
        # At least one value should differ
        differs = any(w[i][j] != noisy[i][j] for i in range(len(w)) for j in range(len(w[0])))
        assert differs

    def test_noise_is_reproducible(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, 0.5], [-0.3, 0.8]]
        n1 = inject_weight_noise(w, seed=123)
        n2 = inject_weight_noise(w, seed=123)
        assert n1 == n2

    def test_different_seeds_differ(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, 0.5]]
        n1 = inject_weight_noise(w, seed=1)
        n2 = inject_weight_noise(w, seed=2)
        assert n1 != n2

    def test_uniform_noise(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, -1.0]]
        noisy = inject_weight_noise(w, noise_model="uniform", seed=42)
        assert len(noisy) == 1
        assert len(noisy[0]) == 2

    def test_lognormal_noise(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, 0.5]]
        noisy = inject_weight_noise(w, noise_model="lognormal", seed=42)
        assert len(noisy[0]) == 2

    def test_noise_profile_creation(self):
        from sc_neurocore.compiler.advanced_features import create_noise_profile

        p = create_noise_profile(
            sigma=0.03,
            target="rain_neuromorphic",
        )
        assert p.noise_model == "gaussian"
        assert p.sigma == 0.03
        assert p.target_platform == "rain_neuromorphic"

    def test_zero_sigma_no_noise(self):
        from sc_neurocore.compiler.advanced_features import inject_weight_noise

        w = [[1.0, -0.5, 0.3]]
        noisy = inject_weight_noise(w, sigma=0.0, seed=42)
        assert noisy == w


# ═══════════════════════════════════════════════════════════════════════
# 15. Pipeline Register Wrapper
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineWrapper:
    """Pipeline register insertion for high-frequency targets."""

    def test_basic_pipeline(self):
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_lif", {"v": "a * b"})
        assert "v_out" in v
        assert "spike_out" in v
        assert "v_reg" in v

    def test_inner_module_instantiation(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_custom", {"v": "a + b"})
        assert "sc_custom core" in v


# ═══════════════════════════════════════════════════════════════════════
# 16. Multi-Target Comparison
# ═══════════════════════════════════════════════════════════════════════


class TestMultiTargetComparison:
    """Compile-once, compare-N-targets."""

    def test_compare_three_targets(self):
        from sc_neurocore.compiler.advanced_features import compare_targets

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "ice40", "loihi2"],
        )
        assert len(results) == 3
        assert results[0].target == "artix7"
        assert results[1].target == "ice40"

    def test_dsp_targets_have_dsps(self):
        from sc_neurocore.compiler.advanced_features import compare_targets

        results = compare_targets(
            {"v": "a * b"},
            ["artix7", "bae_rad750"],
        )
        artix = results[0]
        rad = results[1]
        assert artix.estimated_dsps > 0
        assert rad.estimated_dsps == 0

    def test_format_report(self):
        from sc_neurocore.compiler.advanced_features import (
            compare_targets,
            format_comparison_report,
        )

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        report = format_comparison_report(results)
        assert "Multi-Target" in report
        assert "artix7" in report
        assert "loihi2" in report
        assert "Pipeline" in report

    def test_critical_path_consistent(self):
        from sc_neurocore.compiler.advanced_features import compare_targets

        results = compare_targets(
            {"v": "a * b * c"},
            ["artix7", "ice40"],
        )
        # Same equations → same depth
        assert results[0].critical_path_depth == results[1].critical_path_depth


# ═══════════════════════════════════════════════════════════════════════
# 17. Compilation Summary Report
# ═══════════════════════════════════════════════════════════════════════


class TestCompilationSummary:
    """End-to-end compilation summary generation."""

    def test_summary_contains_sections(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a * b + c"},
            "artix7",
        )
        assert "## Module:" in s
        assert "### Equations" in s
        assert "### Target Platform" in s
        assert "### Fixed-Point Configuration" in s
        assert "### Resource Estimation" in s
        assert "### Pipeline Analysis" in s
        assert "### Applicable Features" in s

    def test_fpga_features(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
        )
        assert "TMR wrapper" in s
        assert "Bitstream encryption" in s

    def test_photonic_features(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "lightmatter_passage",
        )
        assert "MZI weight encoding" in s

    def test_neuromorphic_features(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "loihi2",
        )
        assert "On-chip learning" in s

    def test_verilog_lines_shown(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
            verilog_lines=150,
        )
        assert "150 lines" in s


# ═══════════════════════════════════════════════════════════════════════
# 18. Cross-Feature Integration (E2E)
# ═══════════════════════════════════════════════════════════════════════


class TestCrossFeatureIntegration:
    """End-to-end tests chaining multiple features together."""

    def test_tmr_plus_checksum(self):
        """TMR wrapper + model checksum embedding."""
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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
        from sc_neurocore.compiler.advanced_features import (
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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

import pytest

from sc_neurocore.compiler.platforms import get_profile, list_profile_names


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
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "photonic"
        assert p.data_width > 0
        assert p.fraction < p.data_width

    def test_mzi_dsp_block(self):
        from sc_neurocore.compiler.platforms import get_profile

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
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_wse3_frequency(self):
        from sc_neurocore.compiler.platforms import get_profile

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
        from sc_neurocore.compiler.platforms import get_profile

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
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "neuromorphic"

    def test_brainscales2_wrap(self):
        """BrainScaleS-2 uses wrap overflow (analog domain)."""
        from sc_neurocore.compiler.platforms import get_profile

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
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "fpga"

    def test_rad750_no_dsp(self):
        """RAD750 has no dedicated DSP blocks."""
        from sc_neurocore.compiler.platforms import get_profile

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
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.data_width > 0


class TestTotalProfileCount:
    """Verify total platform coverage."""

    def test_at_least_113_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        names = list_profile_names()
        assert len(names) >= 113, f"Only {len(names)} profiles found"

    def test_10_platform_classes(self):
        from sc_neurocore.compiler.platforms import list_profiles

        classes = {p.platform_class for p in list_profiles()}
        assert len(classes) >= 9

    def test_filter_by_photonic(self):
        from sc_neurocore.compiler.platforms import list_profiles

        photonic = list_profiles(platform_class="photonic")
        assert len(photonic) >= 5

    def test_filter_by_in_memory(self):
        from sc_neurocore.compiler.platforms import list_profiles

        pim = list_profiles(platform_class="in_memory")
        assert len(pim) >= 5


class TestTOMLProfileLoader:
    """TOML-based custom profile registration."""

    def test_load_valid_toml(self, tmp_path):
        from sc_neurocore.compiler.platforms import (
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
        from sc_neurocore.compiler.platforms import load_toml_profile

        with pytest.raises(FileNotFoundError):
            load_toml_profile("/nonexistent/path/chip.toml")

    def test_load_missing_fields(self, tmp_path):
        from sc_neurocore.compiler.platforms import load_toml_profile

        toml = tmp_path / "incomplete.toml"
        toml.write_text("""\
[profile]
name = "incomplete_chip"
vendor = "TestCorp"
""")
        with pytest.raises(ValueError, match="Missing required fields"):
            load_toml_profile(str(toml))

    def test_load_toml_dir(self, tmp_path):
        from sc_neurocore.compiler.platforms import load_toml_profiles_dir

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
        from sc_neurocore.compiler.platforms import load_toml_profiles_dir

        loaded = load_toml_profiles_dir(str(tmp_path))
        assert len(loaded) == 0

    def test_minimal_toml(self, tmp_path):
        """Minimal TOML without optional fields."""
        from sc_neurocore.compiler.platforms import load_toml_profile

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


class TestWeightNoise:
    """Device-variation noise injection for analog robustness."""

    def test_gaussian_noise_changes_weights(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -1.0], [0.5, 0.0]]
        noisy = inject_weight_noise(w, seed=42)
        # At least one value should differ
        differs = any(w[i][j] != noisy[i][j] for i in range(len(w)) for j in range(len(w[0])))
        assert differs

    def test_noise_is_reproducible(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5], [-0.3, 0.8]]
        n1 = inject_weight_noise(w, seed=123)
        n2 = inject_weight_noise(w, seed=123)
        assert n1 == n2

    def test_different_seeds_differ(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5]]
        n1 = inject_weight_noise(w, seed=1)
        n2 = inject_weight_noise(w, seed=2)
        assert n1 != n2

    def test_uniform_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -1.0]]
        noisy = inject_weight_noise(w, noise_model="uniform", seed=42)
        assert len(noisy) == 1
        assert len(noisy[0]) == 2

    def test_lognormal_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, 0.5]]
        noisy = inject_weight_noise(w, noise_model="lognormal", seed=42)
        assert len(noisy[0]) == 2

    def test_noise_profile_creation(self):
        from sc_neurocore.compiler.intelligence import create_noise_profile

        p = create_noise_profile(
            sigma=0.03,
            target="rain_neuromorphic",
        )
        assert p.noise_model == "gaussian"
        assert p.sigma == 0.03
        assert p.target_platform == "rain_neuromorphic"

    def test_zero_sigma_no_noise(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        w = [[1.0, -0.5, 0.3]]
        noisy = inject_weight_noise(w, sigma=0.0, seed=42)
        assert noisy == w


class TestMagnonicPlatforms:
    @pytest.mark.parametrize(
        "name",
        [
            "tum_skyrmion",
            "kaist_spinwave",
            "imec_mtj_reservoir",
        ],
    )
    def test_magnonic(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "magnonic"


class TestOrganicBioelectronic:
    @pytest.mark.parametrize("name", ["cambridge_oect", "linkoping_organic"])
    def test_organic(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "organic_bioelectronic"


class TestRiscVSovereign:
    @pytest.mark.parametrize(
        "name",
        [
            "sifive_x280_ai",
            "esperanto_et_soc",
            "ventana_veyron_ai",
            "tenstorrent_ascalon",
            "andes_ax45mpv",
        ],
    )
    def test_risc_v(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "risc_v_sovereign"


class TestTotalCoverage:
    def test_min_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 175

    def test_min_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 31


class TestFromConstraints:
    def test_basic(self):
        from sc_neurocore.compiler.platforms import (
            HardwareProfile,
            get_profile,
        )

        p = HardwareProfile.from_constraints(
            "test_w10_auto",
            vendor="TestVendor",
            platform_class="custom",
        )
        assert p.data_width >= 8
        assert p.fraction >= 1
        retrieved = get_profile("test_w10_auto")
        assert retrieved.vendor == "TestVendor"

    def test_low_power(self):
        from sc_neurocore.compiler.platforms import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_lowpow",
            max_power_budget_mw=5,
        )
        assert p.data_width == 8

    def test_explicit_width(self):
        from sc_neurocore.compiler.platforms import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_32bit",
            data_width=32,
            fraction=16,
        )
        assert p.data_width == 32
        assert p.fraction == 16


class TestThermodynamicPlatforms(unittest.TestCase):
    def test_extropic_epu(self):
        p = get_profile("extropic_epu")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertEqual(p.vendor, "Extropic")
        self.assertEqual(p.data_width, 8)

    def test_normal_cn101(self):
        p = get_profile("normal_cn101")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertIn("stochastic", p.notes.lower())


class TestProbabilisticPlatforms(unittest.TestCase):
    def test_purdue_pbit(self):
        p = get_profile("purdue_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertEqual(p.vendor, "Purdue")

    def test_tohoku_sot_pbit(self):
        p = get_profile("tohoku_sot_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertIn("SOT", p.notes)


class TestPolaritonPlatforms(unittest.TestCase):
    def test_marvell_polariton(self):
        p = get_profile("marvell_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertEqual(p.vendor, "Marvell")

    def test_stanford_polariton(self):
        p = get_profile("stanford_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertIn("perovskite", p.notes.lower())


class TestMetamaterialPlatforms(unittest.TestCase):
    def test_mit_metamaterial(self):
        p = get_profile("mit_metamaterial")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertEqual(p.vendor, "MIT")

    def test_penn_acoustic_meta(self):
        p = get_profile("penn_acoustic_meta")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertIn("acoustic", p.notes.lower())


class TestExtendedCoverage(unittest.TestCase):
    def test_profile_count_ge_183(self):
        self.assertGreaterEqual(len(list_profile_names()), 183)

    def test_class_count_ge_35(self):
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        self.assertGreaterEqual(len(classes), 35)


def test_wave12_hardware_profiles_exist():
    """Verify Wave 12 platforms are loaded properly."""
    wetware1 = get_profile("cortical_labs_dishbrain")
    assert wetware1.platform_class == "wetware"
    assert wetware1.data_width == 8

    molecular = get_profile("biomemory_dna")
    assert molecular.platform_class == "molecular"
    assert molecular.vendor == "Biomemory"


class TestWave1Profiles:
    """Verify all 12 new hardware profiles are registered."""

    @pytest.mark.parametrize(
        "name",
        [
            "loihi3",
            "northpole",
            "innatera_pulsar",
            "versal_ai_edge",
            "proasic3",
            "trion",
            "titanium",
            "gowin_arora_v",
            "intel_agilex5",
            "nvidia_dla",
            "mediatek_apu",
            "aws_inferentia",
        ],
    )
    def test_profile_exists(self, name):
        """Profile is registered and retrievable."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0
        assert p.fraction >= 0
        assert p.vendor

    def test_total_profiles_at_least_77(self):
        """Total registry should have at least 77 profiles."""
        from sc_neurocore.compiler.platforms import list_profiles

        assert len(list_profiles()) >= 77

    def test_loihi3_is_neuromorphic(self):
        """Loihi 3 should be in the neuromorphic class."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("loihi3")
        assert p.platform_class == "neuromorphic"
        assert p.data_width == 32
        assert p.overflow == "wrap"

    def test_versal_ai_edge_dsp58(self):
        """Versal AI Edge should use DSP58 with 27x24 multiplier."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("versal_ai_edge")
        assert p.dsp_block == "DSP58"
        assert p.dsp_mult_a == 27
        assert p.dsp_mult_b == 24
        assert p.max_freq_mhz == 900


class TestWave1bProfiles:
    """Verify the 7 additional profiles from §1C/1D."""

    @pytest.mark.parametrize(
        "name",
        [
            "qualcomm_nsp",
            "sambanova",
            "cambricon_mlu",
            "superconducting",
            "cim_sram",
            "analog_ai",
            "event_camera",
        ],
    )
    def test_profile_exists(self, name):
        """Profile is registered and retrievable."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0

    def test_total_profiles_at_least_84(self):
        """Registry should now have ≥84 profiles."""
        from sc_neurocore.compiler.platforms import list_profiles

        assert len(list_profiles()) >= 84

    def test_superconducting_is_emerging(self):
        """Superconducting is in the emerging class."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("superconducting")
        assert p.platform_class == "emerging"
        assert p.overflow == "wrap"

    def test_event_camera_matches_dvs(self):
        """Event camera profile matches DVS sensor specs."""
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile("event_camera")
        assert p.vendor == "Prophesee/Sony"
        assert p.data_width == 16


class TestNewPlatformClasses:
    """Verify all 6 new platform classes and 18 new profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "nist_sfq",
            "northrop_aqfp",
            "josephson_jj",
        ],
    )
    def test_superconducting_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "superconducting"
        assert p.max_freq_mhz >= 5000  # GHz-class

    @pytest.mark.parametrize(
        "name",
        [
            "everspin_stt_mram",
            "samsung_sot_mram",
        ],
    )
    def test_spintronic_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "spintronic"

    @pytest.mark.parametrize("name", ["gf_fefet", "sk_hynix_feram"])
    def test_ferroelectric_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "ferroelectric"

    @pytest.mark.parametrize(
        "name",
        [
            "samsung_cgra",
            "qualcomm_npu_cgra",
            "pact_xtensa",
        ],
    )
    def test_cgra_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "cgra"
        assert p.dsp_block  # CGRAs have PE blocks

    @pytest.mark.parametrize(
        "name",
        [
            "tsmc_soic",
            "intel_foveros",
            "amd_3dv",
        ],
    )
    def test_3d_stacked_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "3d_stacked"

    @pytest.mark.parametrize(
        "name",
        [
            "rp2040",
            "esp32_s3",
            "stm32h7",
            "nrf5340",
            "max78000",
        ],
    )
    def test_edge_mcu_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "edge_mcu"

    @pytest.mark.parametrize(
        "name",
        [
            "sifive_x280",
            "qualcomm_ventana",
            "ainekko_rv",
        ],
    )
    def test_riscv_ai_profiles(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_total_profile_count(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 131

    def test_platform_class_count(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 15


class TestFrontierPlatforms:
    """Verify 4 new platform classes and 10 new profiles."""

    @pytest.mark.parametrize(
        "name",
        [
            "finalspark_neuroplatform",
            "cortical_labs_dishbrain",
        ],
    )
    def test_biological(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class in ("biological", "wetware")

    @pytest.mark.parametrize(
        "name",
        [
            "ibm_ecram",
            "samsung_pcram",
            "stanford_ecram",
        ],
    )
    def test_electrochemical(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "electrochemical"

    @pytest.mark.parametrize(
        "name",
        [
            "cerebras_wse3_ws",
            "tesla_dojo3",
            "tachyum_prodigy",
        ],
    )
    def test_wafer_scale(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "wafer_scale"

    @pytest.mark.parametrize(
        "name",
        [
            "aspinity_aml100",
            "renesas_analog_ai",
        ],
    )
    def test_analog_mixed(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        p = get_profile(name)
        assert p.platform_class == "analog_mixed"

    def test_total_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 144

    def test_total_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 19


class TestAutoTargetRecommender:
    def test_basic_recommendation(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"})
        assert len(recs) == 5
        assert recs[0].score >= recs[-1].score

    def test_class_filter(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target(
            {"v": "a * b + c"},
            require_class="neuromorphic",
        )
        for r in recs:
            assert "neuromorphic" in r.rationale

    def test_width_filter(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"}, max_data_width=8)
        from sc_neurocore.compiler.platforms import get_profile

        for r in recs:
            assert get_profile(r.profile_name).data_width <= 8

    def test_top_n(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"}, top_n=3)
        assert len(recs) == 3


class TestFrontierPlatformsW8:
    @pytest.mark.parametrize(
        "name",
        [
            "weebit_reram",
            "crossbar_rram",
            "adesto_cbram",
        ],
    )
    def test_rram(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "rram"

    @pytest.mark.parametrize("name", ["tsmc_cim_n7", "samsung_cim_sf3"])
    def test_sram_cim(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "sram_cim"

    @pytest.mark.parametrize("name", ["intel_horse_ridge", "google_cryo_ctrl"])
    def test_cryo_cmos(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "cryo_cmos"

    @pytest.mark.parametrize(
        "name",
        [
            "microsoft_dna_store",
            "asu_dna_perovskite",
        ],
    )
    def test_dna_molecular(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "dna_molecular"

    @pytest.mark.parametrize("name", ["ibm_qnn", "ionq_trapped_ion"])
    def test_quantum_neuro(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "quantum_neuro"

    def test_total_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 155

    def test_total_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 24


class TestFinalPlatforms:
    @pytest.mark.parametrize("name", ["ayar_teraphy", "intel_cpo"])
    def test_optical_io(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "optical_io"

    @pytest.mark.parametrize("name", ["mit_phononic", "caltech_mems_nn"])
    def test_acoustic(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "acoustic"

    @pytest.mark.parametrize(
        "name",
        [
            "stanford_microfluidic",
            "eth_fluidic_logic",
        ],
    )
    def test_fluidic(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "fluidic"

    @pytest.mark.parametrize(
        "name",
        [
            "bae_rad750_sq",
            "seakr_sbc",
            "vorago_va10820",
            "frontgrade_leon5",
        ],
    )
    def test_space_qualified(self, name):
        from sc_neurocore.compiler.platforms import get_profile

        assert get_profile(name).platform_class == "space_qualified"

    def test_total_profiles(self):
        from sc_neurocore.compiler.platforms import list_profile_names

        assert len(list_profile_names()) >= 164

    def test_total_classes(self):
        from sc_neurocore.compiler.platforms import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 28


class TestTOMLLoader:
    def test_load(self, tmp_path):
        from sc_neurocore.compiler.platforms import load_profiles_from_toml
        from sc_neurocore.compiler.platforms import get_profile

        toml = tmp_path / "custom.toml"
        toml.write_text(
            "[[profile]]\n"
            'name = "test_custom_chip"\n'
            'vendor = "TestVendor"\n'
            'family = "TestFam"\n'
            'platform_class = "custom"\n'
            "data_width = 16\n"
            "fraction = 8\n"
        )
        loaded = load_profiles_from_toml(str(toml))
        assert "test_custom_chip" in loaded
        p = get_profile("test_custom_chip")
        assert p.vendor == "TestVendor"

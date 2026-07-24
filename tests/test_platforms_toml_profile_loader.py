# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTOMLProfileLoader from former test_platforms.py

"""Focused suite: TestTOMLProfileLoader from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


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

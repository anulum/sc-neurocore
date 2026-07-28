# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis yosys json

"""Focused suite: TestYosysJsonParser from former test_studio_synthesis.py."""

from __future__ import annotations

from tests.studio_synthesis_support import *  # noqa: F403


class TestYosysJsonParser:
    def test_parse_empty_design(self, tmp_path):
        data = {"modules": {}}
        json_path = str(tmp_path / "empty.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 0
        assert result["ffs"] == 0
        assert result["cells"] == 0

    def test_parse_with_luts_and_ffs(self, tmp_path):
        data = {
            "modules": {
                "top": {
                    "cells": {
                        "c0": {"type": "SB_LUT4"},
                        "c1": {"type": "SB_LUT4"},
                        "c2": {"type": "SB_DFF"},
                        "c3": {"type": "DSP48"},
                    },
                    "netnames": {"n0": {}, "n1": {}, "n2": {}},
                }
            }
        }
        json_path = str(tmp_path / "design.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 2
        assert result["ffs"] == 1
        assert result["dsps"] == 1
        assert result["cells"] == 4
        assert result["wires"] == 3

    def test_parse_ecp5_and_xilinx_flip_flops(self, tmp_path):
        data = {
            "modules": {
                "top": {
                    "cells": {
                        "ecp5_ff": {"type": "TRELLIS_FF"},
                        "generic_ff": {"type": "$dff"},
                        "xilinx_ff": {"type": "FDRE"},
                    },
                    "netnames": {},
                }
            }
        }
        json_path = str(tmp_path / "target_ffs.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        result = _parse_yosys_json(json_path)

        assert result["ffs"] == 3

    def test_parse_bram_detection(self, tmp_path):
        data = {
            "modules": {
                "mem": {
                    "cells": {
                        "r0": {"type": "SB_RAM256x16"},
                        "r1": {"type": "BRAM_TDP36"},
                    },
                    "netnames": {},
                }
            }
        }
        json_path = str(tmp_path / "mem.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["brams"] == 2

    def test_parse_multi_module(self, tmp_path):
        data = {
            "modules": {
                "a": {"cells": {"c0": {"type": "LUT4"}}, "netnames": {"n": {}}},
                "b": {"cells": {"c0": {"type": "DFF"}}, "netnames": {"n": {}}},
            }
        }
        json_path = str(tmp_path / "multi.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        result = _parse_yosys_json(json_path)
        assert result["luts"] == 1
        assert result["ffs"] == 1
        assert result["cells"] == 2
        assert result["wires"] == 2

    def test_parse_rejects_non_object_modules(self, tmp_path):
        data = {"modules": []}
        json_path = str(tmp_path / "bad_modules.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="'modules' must be an object"):
            _parse_yosys_json(json_path)

    def test_parse_rejects_non_object_cells(self, tmp_path):
        data = {"modules": {"top": {"cells": [], "netnames": {}}}}
        json_path = str(tmp_path / "bad_cells.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="cells' must be an object"):
            _parse_yosys_json(json_path)

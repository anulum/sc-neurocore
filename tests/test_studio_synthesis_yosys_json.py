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


def _write(tmp_path, name, modules):
    json_path = str(tmp_path / name)
    with open(json_path, "w") as f:
        json.dump({"modules": modules}, f)
    return json_path


def _top(*cell_types, nets=0):
    return {
        "cells": {f"c{index}": {"type": cell_type} for index, cell_type in enumerate(cell_types)},
        "netnames": {f"n{index}": {} for index in range(nets)},
    }


class TestYosysJsonParser:
    def test_parse_empty_design(self, tmp_path):
        result, uncounted = _parse_yosys_json(_write(tmp_path, "empty.json", {}), "ice40")
        assert result["luts"] == 0
        assert result["ffs"] == 0
        assert result["cells"] == 0
        assert uncounted == {}

    def test_parse_with_luts_and_ffs(self, tmp_path):
        path = _write(
            tmp_path,
            "design.json",
            {"top": _top("SB_LUT4", "SB_LUT4", "SB_DFFE", "SB_MAC16", nets=3)},
        )
        result, uncounted = _parse_yosys_json(path, "ice40")
        assert result["luts"] == 2
        assert result["ffs"] == 1
        # SB_MAC16 names neither "DSP" nor "MUL": the substring match never
        # counted an iCE40 DSP.
        assert result["dsps"] == 1
        assert result["cells"] == 4
        assert result["wires"] == 3
        assert uncounted == {}

    def test_parse_ecp5_and_xilinx_flip_flops(self, tmp_path):
        ecp5, _ = _parse_yosys_json(_write(tmp_path, "e.json", {"top": _top("TRELLIS_FF")}), "ecp5")
        xilinx, _ = _parse_yosys_json(_write(tmp_path, "x.json", {"top": _top("FDRE")}), "xilinx")
        assert ecp5["ffs"] == 1
        assert xilinx["ffs"] == 1

    def test_a_generic_cell_left_unmapped_is_reported_not_guessed(self, tmp_path):
        result, uncounted = _parse_yosys_json(
            _write(tmp_path, "g.json", {"top": _top("$dff", "SB_LUT4")}), "ice40"
        )
        assert result["ffs"] == 0
        assert uncounted == {"$dff": 1}

    def test_parse_bram_detection(self, tmp_path):
        ice40, _ = _parse_yosys_json(
            _write(tmp_path, "i.json", {"top": _top("SB_RAM40_4K", "SB_RAM40_4KNR")}), "ice40"
        )
        ecp5, _ = _parse_yosys_json(_write(tmp_path, "e.json", {"top": _top("DP16KD")}), "ecp5")
        xilinx, _ = _parse_yosys_json(
            _write(tmp_path, "x.json", {"top": _top("RAMB36E1", "RAMB18E1")}), "xilinx"
        )
        assert ice40["brams"] == 2
        # DP16KD holds no "RAM": ECP5 block RAM was never counted.
        assert ecp5["brams"] == 1
        # In RAMB18 halves: a RAMB36 is two.
        assert xilinx["brams"] == 3

    def test_lut_ram_is_not_block_ram(self, tmp_path):
        xilinx, _ = _parse_yosys_json(
            _write(tmp_path, "x.json", {"top": _top("RAM256X1S", "RAM64X1D")}), "xilinx"
        )
        gowin, uncounted = _parse_yosys_json(
            _write(tmp_path, "g.json", {"top": _top("RAM16SDP4", "RAM16SDP4")}), "gowin"
        )
        assert xilinx["brams"] == 0
        assert xilinx["luts"] == 6
        assert gowin["brams"] == 0
        assert uncounted == {"RAM16SDP4": 2}

    def test_carry_and_wide_multiplexers_are_counted_by_what_they_occupy(self, tmp_path):
        ecp5, _ = _parse_yosys_json(
            _write(tmp_path, "e.json", {"top": _top("CCU2C", "LUT4", "PFUMX")}), "ecp5"
        )
        gowin, _ = _parse_yosys_json(
            _write(tmp_path, "g.json", {"top": _top("LUT4", "ALU", "MUX2_LUT5", "MUX2_LUT8")}),
            "gowin",
        )
        assert ecp5["luts"] == 3
        assert gowin["luts"] == 2

    def test_the_cell_library_is_not_the_design(self, tmp_path):
        library = {
            "attributes": {"blackbox": "00000000000000000000000000000001"},
            **_top("$specify2", "$specrule", nets=4),
        }
        path = _write(
            tmp_path, "lib.json", {"SB_RAM40_4K": library, "top": _top("SB_LUT4", nets=1)}
        )
        result, uncounted = _parse_yosys_json(path, "ice40")
        assert result["cells"] == 1
        assert result["wires"] == 1
        assert uncounted == {}

    def test_parse_multi_module(self, tmp_path):
        path = _write(tmp_path, "multi.json", {"a": _top("LUT4", nets=1), "b": _top("DFF", nets=1)})
        result, _ = _parse_yosys_json(path, "gowin")
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
            _parse_yosys_json(json_path, "ice40")

    def test_parse_rejects_non_object_cells(self, tmp_path):
        data = {"modules": {"top": {"cells": [], "netnames": {}}}}
        json_path = str(tmp_path / "bad_cells.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="cells' must be an object"):
            _parse_yosys_json(json_path, "ice40")

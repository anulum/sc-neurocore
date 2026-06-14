# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the ORCA NMR parameter extractor

"""Unit and CLI tests for ``tools/quantum/extract_orca_nmr.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/quantum/extract_orca_nmr.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_orca_nmr", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _full_output(*, terminated: bool = True, omit_shielding: bool = False) -> str:
    shielding = (
        ""
        if omit_shielding
        else """\
--------------------------------
CHEMICAL SHIELDING SUMMARY (ppm)
--------------------------------


  Nucleus  Element    Isotropic     Anisotropy
  -------  -------  ------------   ------------
      9       P          274.939        114.525
     10       P          274.682        112.897
      0       Ca        1148.269        105.112

"""
    )
    tail = ""
    if terminated:
        tail = """\
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 1 hours 4 minutes 5 seconds 414 msec
"""
    return f"""\
                         Program Version 6.1.1  -  RELEASE   -
|  1> ! B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 NMR
|  7> * xyzfile 0 1 input.xyz
General Settings:
 Hartree-Fock type      HFTyp           .... RHF
 Total Charge           Charge          ....    0
 Multiplicity           Mult            ....    1
 Number of Electrons    NEL             ....  462
 Basis Dimension        Dim             .... 1290
FINAL SINGLE POINT ENERGY     -9954.022370142919
{shielding}
-----------------------------------------------------------------------------
                SUMMARY OF ISOTROPIC COUPLING CONSTANTS J (Hz)
-----------------------------------------------------------------------------
                  9 P       10 P       11 P
      9 P        0.000      0.138      0.000
     10 P        0.138      0.000      0.914
     11 P        0.000      0.914      0.000

NMR spin-spin coupling calculation done in  15.8 sec
    Number of nuclear pairs to calculate something:        2
{tail}"""


def _write_output(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_extract_parses_nmr_sections(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())

    record = tool.extract_orca_nmr(out, required_shielding_elements=("P", "Ca"))

    assert record["schema_version"] == tool.SCHEMA_VERSION
    assert record["final_single_point_energy_eh"] == pytest.approx(-9954.022370142919)
    assert record["termination"]["normal_termination"] is True
    assert record["termination"]["total_run_time_seconds"] == pytest.approx(3845.414)
    assert record["run_settings"]["program_version"] == "6.1.1"
    assert record["run_settings"]["route_line"].endswith("DefGrid3 NMR")
    assert record["run_settings"]["charge"] == 0
    assert record["run_settings"]["multiplicity"] == 1
    assert record["run_settings"]["basis_dimension"] == 1290


def test_extract_shielding_grouped_by_element(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())

    shielding = tool.extract_orca_nmr(out)["chemical_shielding"]

    assert shielding["nucleus_count"] == 3
    assert shielding["by_element"]["P"][0] == {
        "atom_index": 9,
        "element": "P",
        "isotropic_ppm": 274.939,
        "anisotropy_ppm": 114.525,
    }
    assert shielding["by_element"]["Ca"][0]["isotropic_ppm"] == pytest.approx(1148.269)


def test_extract_spin_spin_coupling_matrix_and_pairs(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())

    coupling = tool.extract_orca_nmr(out)["spin_spin_coupling"]

    assert coupling["reported_pair_count"] == 2
    assert coupling["matrix_labels"] == ["9P", "10P", "11P"]
    assert coupling["matrix_hz"]["9P"]["10P"] == pytest.approx(0.138)
    assert coupling["matrix_hz"]["10P"]["11P"] == pytest.approx(0.914)
    assert coupling["nonzero_pairs"] == [
        {"atom_a": 9, "element_a": "P", "atom_b": 10, "element_b": "P", "j_iso_hz": 0.138},
        {"atom_a": 10, "element_a": "P", "atom_b": 11, "element_b": "P", "j_iso_hz": 0.914},
    ]


def test_provenance_records_sha256_and_absolute_paths(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())
    extra = tmp_path / "input.xyz"
    extra.write_text("3\n\nP 0 0 0\n", encoding="utf-8")

    sources = tool.extract_orca_nmr(out, extra_sources=[extra])["provenance"]["sources"]

    assert [source["role"] for source in sources] == ["orca_output", "provenance"]
    assert sources[0]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert sources[1]["sha256"] == hashlib.sha256(extra.read_bytes()).hexdigest()
    assert Path(sources[0]["path"]).is_absolute()


def test_serialised_json_is_deterministic(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())

    rendered = tool.serialise(tool.extract_orca_nmr(out))

    assert rendered == tool.serialise(tool.extract_orca_nmr(out))
    assert rendered.endswith("\n")
    json.loads(rendered)


def test_missing_termination_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output(terminated=False))

    with pytest.raises(tool.OrcaNmrExtractionError, match="normal-termination"):
        tool.extract_orca_nmr(out)


def test_missing_shielding_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output(omit_shielding=True))

    with pytest.raises(tool.OrcaNmrExtractionError, match="CHEMICAL SHIELDING"):
        tool.extract_orca_nmr(out)


def test_missing_required_element_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "nmr.out", _full_output())

    with pytest.raises(tool.OrcaNmrExtractionError, match="missing.*H"):
        tool.extract_orca_nmr(out, required_shielding_elements=("H",))


def test_cli_writes_json(tmp_path: Path) -> None:
    out = _write_output(tmp_path / "nmr.out", _full_output())
    output = tmp_path / "params.json"

    completed = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert output.is_file()
    assert (
        json.loads(output.read_text(encoding="utf-8"))["chemical_shielding"]["nucleus_count"] == 3
    )

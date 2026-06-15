# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the ORCA EPR/HFC parameter extractor

"""Unit and CLI tests for ``tools/quantum/extract_orca_params.py``.

The fixtures are compact synthetic ORCA fragments that reproduce the exact
line layout of an ORCA 6.1 ``EPRNMR`` output (gtensor + hyperfine) for one
phosphorus and one calcium nucleus, so the parser is exercised without
shipping a multi-hundred-kilobyte real output blob.
"""

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
TOOL = REPO / "tools/quantum/extract_orca_params.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_orca_params", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# A compact but layout-faithful ORCA EPR/HFC output fragment.
_G_MATRIX_BLOCK = """\
ELECTRONIC G-MATRIX
-------------------

The g-matrix:
              2.0251815   -0.0060895    0.0127598
             -0.0063840    2.0679999   -0.0347135
              0.0131680   -0.0328507    2.0382556
 Breakdown of the contributions
 gel          2.0023193    2.0023193    2.0023193
             ----------   ----------   ----------
 g(tot)       2.0114962    2.0276218    2.0923195 iso=  2.0438125
 Delta-g      0.0091769    0.0253025    0.0900002 iso=  0.0414932
"""

_P_BLOCK = """\
 Nucleus   9P : A  : Isotope=   31 I=  0.5 P=216.1834 MHz/au**3
                HFC: iso  =YES dip=YES orb=YES gauge=YES

 Total HFC matrix (all values in MHz):
               -33.5215              -0.5928              -0.8634
                -0.4863             -30.3245               0.9557
                -0.8853               1.0367             -31.3777

 A(FC)         -31.8049             -31.8049             -31.8049
 A(SD)           2.3221              -0.1421              -2.1800
 A(ORB+DIA)     -0.0292               0.0768               0.1435    A(PC) =    0.0637
 A(ORB)         -0.0297               0.0765               0.1431    A(PC) =    0.0633
 A(DIA)          0.0005               0.0003               0.0004    A(PC) =    0.0004
             ----------           ----------           ----------
 A(Tot)        -29.5120             -31.8702             -33.8414    A(iso)=  -31.7412
"""

_CA_BLOCK = """\
 Nucleus   0Ca: A  : Isotope=   43 I=  3.5 P=-35.9513 MHz/au**3
                HFC: iso  =YES dip=YES orb= NO gauge= NO

 Total HFC matrix (all values in MHz):
                 1.0548              -0.1123              -0.1961
                -0.1123               1.3845              -0.0915
                -0.1961              -0.0915               1.2675

 A(FC)           1.2356               1.2356               1.2356
 A(SD)          -0.3397               0.1405               0.1992
             ----------           ----------           ----------
 A(Tot)          0.8959               1.3761               1.4348    A(iso)=    1.2356
"""


def _full_output(
    *,
    g_block: str = _G_MATRIX_BLOCK,
    p_block: str = _P_BLOCK,
    ca_block: str = _CA_BLOCK,
    final_energy_line: str = "FINAL SINGLE POINT ENERGY     -9953.726192774189\n",
    run_time_line: str = ("TOTAL RUN TIME: 0 days 4 hours 0 minutes 52 seconds 740 msec\n"),
    terminated: bool = True,
) -> str:
    parts = [
        "                         Program Version 6.1.1  -  RELEASE   -\n",
        "|  1> ! UKS B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 SP\n",
        "|  7> * xyzfile 1 2 input.xyz\n",
        "General Settings:\n",
        " Hartree-Fock type      HFTyp           .... UHF\n",
        " Total Charge           Charge          ....    1\n",
        " Multiplicity           Mult            ....    2\n",
        " Number of Electrons    NEL             ....  461\n",
        " Basis Dimension        Dim             .... 1290\n",
        final_energy_line,
        g_block,
        "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (15 nuclei)\n",
        "Energy             : -9953.4792582384379784 Eh\n",
        p_block,
        ca_block,
    ]
    if terminated:
        parts.append("                             ****ORCA TERMINATED NORMALLY****\n")
        parts.append(run_time_line)
    return "".join(parts)


def _write_output(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #


def test_extract_parses_every_section(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    record = tool.extract_orca_parameters(out)

    assert record["schema_version"] == tool.SCHEMA_VERSION
    assert record["final_single_point_energy_eh"] == pytest.approx(-9953.726192774189)
    assert record["termination"]["normal_termination"] is True
    assert record["termination"]["total_run_time_seconds"] == pytest.approx(14452.74)
    assert (
        record["termination"]["total_run_time_text"]
        == "0 days 4 hours 0 minutes 52 seconds 740 msec"
    )

    settings = record["run_settings"]
    assert settings["program_version"] == "6.1.1"
    assert settings["route_line"] == "UKS B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 SP"
    assert settings["hartree_fock_type"] == "UHF"
    assert settings["charge"] == 1
    assert settings["multiplicity"] == 2
    assert settings["number_of_electrons"] == 461
    assert settings["basis_dimension"] == 1290
    assert settings["property_module_energy_eh"] == pytest.approx(-9953.479258238438)


def test_extract_g_tensor_values(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    g = tool.extract_orca_parameters(out)["g_tensor"]

    assert g["g_matrix"][0] == [2.0251815, -0.0060895, 0.0127598]
    assert g["g_matrix"][2] == [0.0131680, -0.0328507, 2.0382556]
    assert g["g_principal"] == [2.0114962, 2.0276218, 2.0923195]
    assert g["g_isotropic"] == pytest.approx(2.0438125)
    assert g["delta_g_principal"] == [0.0091769, 0.0253025, 0.0900002]
    assert g["delta_g_isotropic"] == pytest.approx(0.0414932)


def test_extract_phosphorus_hyperfine_includes_orbital_term(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    by_element = tool.extract_orca_parameters(out)["hyperfine"]["by_element"]
    assert by_element["P"][0] == {
        "atom_index": 9,
        "element": "P",
        "isotope": 31,
        "spin_quantum_number": 0.5,
        "prefactor_mhz_per_au3": 216.1834,
        "a_fc_isotropic_mhz": -31.8049,
        "a_sd_principal_mhz": [2.3221, -0.1421, -2.1800],
        "a_orb_principal_mhz": [-0.0297, 0.0765, 0.1431],
        "a_orb_isotropic_mhz": 0.0633,
        "a_tot_principal_mhz": [-29.5120, -31.8702, -33.8414],
        "a_isotropic_mhz": -31.7412,
    }


def test_extract_calcium_hyperfine_has_no_orbital_term(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    ca = tool.extract_orca_parameters(out)["hyperfine"]["by_element"]["Ca"][0]
    assert ca["element"] == "Ca"
    assert ca["isotope"] == 43
    assert ca["spin_quantum_number"] == 3.5
    assert ca["prefactor_mhz_per_au3"] == -35.9513
    assert ca["a_fc_isotropic_mhz"] == 1.2356
    assert ca["a_sd_principal_mhz"] == [-0.3397, 0.1405, 0.1992]
    assert ca["a_orb_principal_mhz"] is None
    assert ca["a_orb_isotropic_mhz"] is None
    assert ca["a_isotropic_mhz"] == 1.2356


def test_nucleus_count_and_element_grouping(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    hyperfine = tool.extract_orca_parameters(out)["hyperfine"]
    assert hyperfine["nucleus_count"] == 2
    assert sorted(hyperfine["by_element"]) == ["Ca", "P"]


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #


def test_provenance_records_sha256_and_absolute_paths(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())
    extra = tmp_path / "REPRO.md"
    extra.write_text("reproducibility log\n", encoding="utf-8")

    sources = tool.extract_orca_parameters(out, extra_sources=[extra])["provenance"]["sources"]

    assert [s["role"] for s in sources] == ["orca_output", "provenance"]
    assert sources[0]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert sources[1]["sha256"] == hashlib.sha256(extra.read_bytes()).hexdigest()
    assert Path(sources[0]["path"]).is_absolute()
    assert sources[0]["name"] == "run.out"


def test_missing_extra_source_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    with pytest.raises(tool.OrcaExtractionError, match="Provenance source file not found"):
        tool.extract_orca_parameters(out, extra_sources=[tmp_path / "absent.md"])


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #


def test_serialised_json_is_deterministic(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    record = tool.extract_orca_parameters(out)
    assert tool.serialise(record) == tool.serialise(record)
    assert tool.serialise(record).endswith("\n")
    json.loads(tool.serialise(record))  # round-trips as valid JSON


# --------------------------------------------------------------------------- #
# Fail-closed error paths                                                      #
# --------------------------------------------------------------------------- #


def test_missing_termination_marker_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(terminated=False))

    with pytest.raises(tool.OrcaExtractionError, match="normal-termination marker not found"):
        tool.extract_orca_parameters(out)


def test_missing_run_time_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    text = _full_output().replace(
        "TOTAL RUN TIME: 0 days 4 hours 0 minutes 52 seconds 740 msec\n", ""
    )
    out = _write_output(tmp_path / "run.out", text)

    with pytest.raises(tool.OrcaExtractionError, match="TOTAL RUN TIME"):
        tool.extract_orca_parameters(out)


def test_missing_final_energy_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    text = _full_output(final_energy_line="")
    out = _write_output(tmp_path / "run.out", text)

    with pytest.raises(tool.OrcaExtractionError, match="FINAL SINGLE POINT ENERGY"):
        tool.extract_orca_parameters(out)


def test_missing_g_matrix_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(g_block=""))

    with pytest.raises(tool.OrcaExtractionError, match="ELECTRONIC G-MATRIX"):
        tool.extract_orca_parameters(out)


def test_incomplete_g_matrix_block_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    truncated = "ELECTRONIC G-MATRIX\nThe g-matrix: \n   2.0  0.0  0.0\n"
    out = _write_output(tmp_path / "run.out", _full_output(g_block=truncated))

    with pytest.raises(tool.OrcaExtractionError, match="incomplete or malformed"):
        tool.extract_orca_parameters(out)


def test_required_element_absent_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(ca_block=""))

    with pytest.raises(tool.OrcaExtractionError, match="missing.*Ca"):
        tool.extract_orca_parameters(out)


def test_custom_required_elements_pass_when_present(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(ca_block=""))

    record = tool.extract_orca_parameters(out, required_elements=("P",))
    assert list(record["hyperfine"]["by_element"]) == ["P"]


def test_nucleus_block_missing_a_tot_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    broken_p = _P_BLOCK.replace(
        " A(Tot)        -29.5120             -31.8702             -33.8414    A(iso)=  -31.7412\n",
        "",
    )
    out = _write_output(tmp_path / "run.out", _full_output(p_block=broken_p))

    with pytest.raises(
        tool.OrcaExtractionError, match="missing A\\(FC\\), A\\(SD\\) or A\\(Tot\\)"
    ):
        tool.extract_orca_parameters(out)


def test_missing_input_file_raises(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(tool.OrcaExtractionError, match="ORCA output file not found"):
        tool.extract_orca_parameters(tmp_path / "absent.out")


def test_parse_hyperfine_without_section_header_raises() -> None:
    tool = _load_tool()
    with pytest.raises(tool.OrcaExtractionError, match="HYPERFINE STRUCTURE section not found"):
        tool.parse_hyperfine("no property section here\n")


def test_parse_hyperfine_section_without_nuclei_raises() -> None:
    tool = _load_tool()
    text = "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (0 nuclei)\nno nucleus blocks\n"
    with pytest.raises(tool.OrcaExtractionError, match="No hyperfine nucleus blocks found"):
        tool.parse_hyperfine(text)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def test_cli_writes_deterministic_json_file(tmp_path: Path) -> None:
    out = _write_output(tmp_path / "run.out", _full_output())
    target = tmp_path / "nested" / "params.json"

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    first = target.read_bytes()
    subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )
    assert target.read_bytes() == first
    payload = json.loads(first)
    assert payload["hyperfine"]["nucleus_count"] == 2


def test_cli_stdout_when_no_output(tmp_path: Path) -> None:
    out = _write_output(tmp_path / "run.out", _full_output())

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["final_single_point_energy_eh"] == pytest.approx(-9953.726192774189)


def test_cli_fails_closed_without_writing(tmp_path: Path) -> None:
    out = _write_output(tmp_path / "run.out", _full_output(terminated=False))
    target = tmp_path / "params.json"

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 1
    assert not target.exists()
    assert "normal-termination marker not found" in result.stderr


# --------------------------------------------------------------------------- #
# In-process main() — covers the CLI body for coverage and argument wiring     #
# --------------------------------------------------------------------------- #


def test_main_writes_file_and_extra_source(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())
    extra = tmp_path / "REPRO.md"
    extra.write_text("log\n", encoding="utf-8")
    target = tmp_path / "out" / "params.json"

    code = tool.main(["--input", str(out), "--output", str(target), "--source", str(extra)])

    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [s["role"] for s in payload["provenance"]["sources"]] == [
        "orca_output",
        "provenance",
    ]


def test_main_writes_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    code = tool.main(["--input", str(out)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["g_tensor"]["g_isotropic"] == pytest.approx(2.0438125)


def test_main_custom_required_element_override(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(ca_block=""))

    code = tool.main(["--input", str(out), "--require-element", "P"])

    assert code == 0


def test_main_returns_one_and_writes_nothing_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(terminated=False))
    target = tmp_path / "params.json"

    code = tool.main(["--input", str(out), "--output", str(target)])

    assert code == 1
    assert not target.exists()
    assert "normal-termination marker not found" in capsys.readouterr().err

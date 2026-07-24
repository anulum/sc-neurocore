# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (fail_closed) from former test_extract_orca_params.py

from __future__ import annotations

from extract_orca_params_support import *  # noqa: F403

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

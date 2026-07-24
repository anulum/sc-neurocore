# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (extract_parse) from former test_extract_orca_params.py

from __future__ import annotations

from extract_orca_params_support import *  # noqa: F403

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


def test_serialised_json_is_deterministic(tmp_path: Path) -> None:
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    record = tool.extract_orca_parameters(out)
    assert tool.serialise(record) == tool.serialise(record)
    assert tool.serialise(record).endswith("\n")
    json.loads(tool.serialise(record))  # round-trips as valid JSON

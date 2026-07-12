# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli nir tests

"""Exercise cli nir behaviour through the public CLI."""

from __future__ import annotations

import json
from pathlib import Path
import types
from unittest import mock

import pytest

from sc_neurocore.ir import SCNIR_SCHEMA_VERSION, validate_scnir_dict
from tests.cli_nir_test_support import (
    _aer_equivalence_testbench,
    _aer_fixed_point_reference,
    _aer_lif_nir_graph,
    _dense_lif_nir_graph,
    _direct_equivalence_testbench,
    _mixed_aer_li_lif_nir_graph,
    _mixed_equivalence_testbench,
    _mixed_fixed_point_reference,
    _mixed_li_lif_nir_graph,
    _nested_multiport_multioutput_lif_nir_graph,
    _nested_single_port_lif_nir_graph,
    _parse_aer_equivalence_stdout,
    _parse_direct_equivalence_stdout,
    _parse_mixed_equivalence_stdout,
    _parse_recurrent_equivalence_stdout,
    _recurrent_equivalence_testbench,
    _recurrent_fixed_point_reference,
    _recurrent_lif_nir_graph,
    _simulate_manifest_source,
    _simulate_network_bundle,
    _simulate_network_with_testbench,
    _simulate_single_source_module,
    _small_direct_fixed_point_reference,
    _small_lif_nir_graph,
)
from tests.cli_test_support import fake_module, run_cli


class TestCompileNirCommand:
    """Tests for `sc-neurocore compile-nir ...` exported artefacts."""

    def test_compile_nir_requires_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing graph path returns actionable command usage."""
        assert run_cli("compile-nir") == 1
        assert "requires a model file" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (("--data-width", "1"), "data-width > 1"),
            (("--data-width", "16", "--fraction", "16"), "fraction < data-width"),
        ],
    )
    def test_compile_nir_rejects_invalid_precision(
        self,
        arguments: tuple[str, ...],
        message: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Invalid fixed-point layouts fail before importing the graph."""
        assert run_cli("compile-nir", str(tmp_path / "model.nir"), *arguments) == 1
        assert message in capsys.readouterr().out

    def test_compile_nir_rejects_unknown_extension(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Only the NIR importer formats are accepted."""
        assert run_cli("compile-nir", str(tmp_path / "model.pt")) == 1
        assert "supports .nir and .onnx" in capsys.readouterr().out

    def test_compile_nir_reports_folded_web_metrics_without_fpga_area(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-FPGA targets retain folded metrics and warnings without inventing area data."""
        import sc_neurocore.ir as ir
        import sc_neurocore.nir_bridge as bridge

        model_path = tmp_path / "model.nir"
        model_path.write_bytes(b"fixture")
        output = tmp_path / "compiled"
        fake_nir = fake_module("nir", read=lambda _path: object())
        network = types.SimpleNamespace(topo_order=["input", "lif"])
        neuron_graph = types.SimpleNamespace(
            total_neurons=1,
            total_synapses=0,
            neuron_types={"lif"},
        )
        folded_metrics = types.SimpleNamespace(
            populations=1,
            pe_instances=1,
            shared_multipliers=2,
            state_ram_bits=16,
            cycles_per_tick=2,
            direct_neuron_instances=1,
            as_dict=lambda: {"populations": 1, "pe_instances": 1},
        )
        stream = types.SimpleNamespace(signal_kind="spike")
        document = types.SimpleNamespace(
            streams=[stream],
            hierarchy=[types.SimpleNamespace(ports=[object()])],
        )
        manifest_entry = types.SimpleNamespace(as_dict=lambda: {"stream_id": "pop.lif.spike"})
        external_input = types.SimpleNamespace(as_dict=lambda: {"node": "input"})
        result = types.SimpleNamespace(
            interconnect="folded",
            neuron_modules={"lif": "module sc_nir_lif; endmodule\n"},
            scnir_source_modules={"source_fixture": "module source_fixture; endmodule\n"},
            scnir_hierarchy_modules={"hier_fixture": "module hier_fixture; endmodule\n"},
            folded_metrics=folded_metrics,
            top_module="module folded_web; endmodule\n",
            weight_rom="module sc_nir_weight_rom; endmodule\n",
            scnir_document=document,
            scnir_external_inputs=[external_input],
            scnir_source_manifest=[manifest_entry],
            module_name="folded_web",
            q_format="Q8.8",
            total_neurons=1,
            total_synapses=0,
            warnings=["fixture warning"],
        )
        monkeypatch.setattr(bridge, "from_nir", lambda _graph, *, dt: network)
        monkeypatch.setattr(bridge, "from_scnetwork", lambda _network, *, dt: neuron_graph)
        monkeypatch.setattr(bridge, "compile_network_to_fpga", lambda *_args, **_kwargs: result)
        monkeypatch.setattr(
            ir,
            "write_scnir",
            lambda path, _document: Path(path).write_text("{}\n", encoding="utf-8"),
        )

        with mock.patch.dict("sys.modules", {"nir": fake_nir}):
            assert (
                run_cli(
                    "compile-nir",
                    str(model_path),
                    "--target",
                    "web",
                    "--interconnect",
                    "folded",
                    "--module-name",
                    "folded_web",
                    "--output",
                    str(output),
                )
                == 0
            )

        metrics = json.loads((output / "folded_metrics.json").read_text(encoding="utf-8"))
        assert "area_estimate" not in metrics
        assert (output / "hier_fixture.v").is_file()
        assert "fixture warning" in capsys.readouterr().out

    def test_compile_nir_writes_scnir_source_bundle_and_simulates_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())

        out_dir = tmp_path / "compiled"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "fixture_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "66",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        assert (out_dir / "fixture_net.v").exists()
        assert (out_dir / "sc_nir_lif.v").exists()
        assert (out_dir / "sc_nir_weight_rom.v").exists()

        manifest_path = out_dir / "scnir_source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "sc-neurocore.scnir.hdl-sources.v0.2"
        assert len(manifest["sources"]) == 2

        first = manifest["sources"][0]
        assert first["stream_id"] == "pop.lif.spike"
        assert first["source_kind"] == "sobol16"
        assert first["seed"] == 66
        assert first["bitstream_length"] == 512
        assert first["sobol_dimension"] == 1

        source_path = out_dir / f"{first['module_name']}.v"
        source_verilog = source_path.read_text(encoding="utf-8")
        assert "module " + first["module_name"] in source_verilog
        assert "localparam [15:0] SEED = 16'h0042;" in source_verilog

        stdout = _simulate_single_source_module(first["module_name"], source_verilog, tmp_path)
        assert "sample0 8042 1 1" in stdout
        assert f"{first['module_name']}.v" in capsys.readouterr().out

    def test_compile_nir_folded_interconnect_reports_metrics(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "folded_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())

        out_dir = tmp_path / "folded"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "folded_net",
            "--interconnect",
            "folded",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "Interconnect: folded" in out
        assert "Folded datapath: 1 population(s), 1 PE" in out
        assert "collapses 2 direct neuron instances" in out
        # The folded resource counts are mapped onto a pre-synthesis area estimate.
        assert "Folded area (~est. ice40)" in out
        assert "LUTs" in out and "DSP" in out
        # The shared datapath PE module is emitted alongside the top.
        assert (out_dir / "folded_net.v").exists()
        assert (out_dir / "sc_nir_lif_pe.v").exists()
        # The source-handoff manifest keeps its versioned schema (no metrics pollution).
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "folded"
        assert "folded_metrics" not in manifest
        # Folded metrics are persisted to their own machine-readable artefact.
        metrics = json.loads((out_dir / "folded_metrics.json").read_text(encoding="utf-8"))
        assert metrics["pe_instances"] == 1
        assert metrics["neurons"] == 2
        assert metrics["direct_neuron_instances"] == 2
        assert metrics["shared_multipliers"] == 2
        assert metrics["populations"] == 1
        # The pre-synthesis area estimate is persisted alongside the raw counts.
        area = metrics["area_estimate"]
        assert area["target"] == "ice40"
        assert area["latency_cycles"] == metrics["cycles_per_tick"]
        assert area["total_luts"] > 0
        assert "fits_on_target" in area

    def test_compile_nir_writes_valid_dense_scnir_document(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "dense_fixture.nir"
        nir.write(str(model_path), _dense_lif_nir_graph())

        out_dir = tmp_path / "dense_compiled"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "dense_fixture_net",
            "--T",
            "768",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "101",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        scnir_path = out_dir / "scnir_document.json"
        payload = json.loads(scnir_path.read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert payload["schema_version"] == SCNIR_SCHEMA_VERSION
        assert {stream["bitstream_length"] for stream in payload["streams"]} == {768}
        assert {stream["stream_id"] for stream in payload["streams"]} == {
            "pop.lif1.spike",
            "pop.lif2.spike",
            "conn.input_to_lif1.weight",
            "conn.lif1_to_lif2.weight",
        }
        assert {stream["signal_kind"] for stream in payload["streams"]} == {"spike", "weight"}
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert [row["stream_id"] for row in manifest["sources"]] == [
            stream["stream_id"] for stream in payload["streams"]
        ]
        assert "scnir_document.json" in capsys.readouterr().out

    def test_compile_nir_records_aer_interconnect_in_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "aer_fixture.nir"
        nir.write(str(model_path), _aer_lif_nir_graph())

        out_dir = tmp_path / "aer_compiled"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "aer_fixture_net",
            "--T",
            "384",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "11",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        top_module = (out_dir / "aer_fixture_net.v").read_text(encoding="utf-8")
        assert "Interconnect: weighted event routing" in top_module
        assert "localparam integer AER_SRC_COUNT = 67;" in top_module
        assert "$signed({{(ACC_WIDTH - DATA_WIDTH){Q_MAX[DATA_WIDTH - 1]}}, Q_MAX})" in top_module
        assert "$signed({{(ACC_WIDTH - DATA_WIDTH){Q_MIN[DATA_WIDTH - 1]}}, Q_MIN})" in top_module

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["q_format"] == "Q8.8"
        assert manifest["total_neurons"] == 67
        assert manifest["total_synapses"] == 390
        assert manifest["scnir_stream_count"] == 4

        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert [row["stream_id"] for row in manifest["sources"]] == [
            stream["stream_id"] for stream in payload["streams"]
        ]
        assert "Interconnect: aer" in capsys.readouterr().out

    def test_compile_nir_records_mixed_signal_summary_in_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_signal_fixture.nir"
        nir.write(str(model_path), _mixed_li_lif_nir_graph())

        out_dir = tmp_path / "mixed_signal_compiled"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "mixed_signal_fixture_net",
            "--T",
            "640",
            "--source-kind",
            "sobol",
            "--base-seed",
            "91",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert {stream["stream_id"]: stream["signal_kind"] for stream in payload["streams"]} == {
            "pop.li.state": "analogue_state",
            "pop.lif.spike": "spike",
            "conn.input_to_li.weight": "weight",
            "conn.li_to_lif.weight": "weight",
        }

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_signal_kinds"] == {
            "analogue_state": 1,
            "spike": 1,
            "weight": 2,
        }
        assert [row["signal_kind"] for row in manifest["sources"]] == [
            stream["signal_kind"] for stream in payload["streams"]
        ]
        assert "Interconnect: direct" in capsys.readouterr().out

    def test_compile_nir_records_mixed_aer_routing_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_aer_fixture.nir"
        nir.write(str(model_path), _mixed_aer_li_lif_nir_graph())

        out_dir = tmp_path / "mixed_aer_compiled"
        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "mixed_aer_fixture_net",
            "--T",
            "896",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "123",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        top_module = (out_dir / "mixed_aer_fixture_net.v").read_text(encoding="utf-8")
        assert "Interconnect: weighted event routing" in top_module
        assert "localparam integer AER_SRC_COUNT = 66;" in top_module
        assert "p0_n0_v * 16'sh0080" in top_module
        assert "p0_n1_v * 16'shffc0" in top_module

        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert {stream["stream_id"]: stream["signal_kind"] for stream in payload["streams"]} == {
            "pop.li.state": "analogue_state",
            "pop.lif1.spike": "spike",
            "pop.lif2.spike": "spike",
            "conn.input_to_li.weight": "weight",
            "conn.li_to_lif1.weight": "weight",
            "conn.lif1_to_lif2.weight": "weight",
        }

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["scnir_signal_kinds"] == {
            "analogue_state": 1,
            "spike": 2,
            "weight": 3,
        }
        assert manifest["scnir_signal_routes"] == {
            "analogue_state": "direct_mac",
            "spike": "weighted_event_aer",
            "weight": "stochastic_source_module",
        }
        assert "Interconnect: aer" in capsys.readouterr().out

    def test_compile_nir_cosimulates_sources_across_interconnects(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        cases = [
            (
                "direct_sobol",
                _small_lif_nir_graph(),
                "sobol",
                66,
                "direct",
                "pop.lif.spike",
            ),
            (
                "aer_lfsr",
                _aer_lif_nir_graph(),
                "lfsr",
                11,
                "aer",
                "pop.lif1.spike",
            ),
            (
                "recurrent_lfsr",
                _recurrent_lif_nir_graph(),
                "lfsr",
                41,
                "direct",
                "conn.lif_to_lif.weight",
            ),
        ]

        for name, graph, source_kind, base_seed, expected_interconnect, stream_id in cases:
            model_path = tmp_path / f"{name}.nir"
            nir.write(str(model_path), graph)
            out_dir = tmp_path / f"{name}_compiled"

            rc = run_cli(
                "compile-nir",
                str(model_path),
                "--module-name",
                f"{name}_net",
                "--T",
                "512",
                "--source-kind",
                source_kind,
                "--base-seed",
                str(base_seed),
                "-o",
                str(out_dir),
            )

            assert rc == 0
            manifest = json.loads(
                (out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["interconnect"] == expected_interconnect
            row = next(item for item in manifest["sources"] if item["stream_id"] == stream_id)
            assert row["source_kind"] == f"{source_kind}16"
            sim_dir = tmp_path / f"{name}_sim"
            sim_dir.mkdir()
            _simulate_manifest_source(out_dir, row, sim_dir)

        output = capsys.readouterr().out
        assert "Interconnect: direct" in output
        assert "Interconnect: aer" in output

    def test_compile_nir_cosimulates_complete_networks_across_interconnects(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir = pytest.importorskip("nir")
        cases = [
            (
                "direct_top_sobol",
                _small_lif_nir_graph(),
                "sobol",
                66,
                2,
                2,
                "direct",
            ),
            (
                "aer_top_lfsr",
                _aer_lif_nir_graph(),
                "lfsr",
                11,
                4,
                67,
                "aer",
            ),
            (
                "recurrent_top_lfsr",
                _recurrent_lif_nir_graph(),
                "lfsr",
                41,
                2,
                2,
                "direct",
            ),
        ]

        for name, graph, source_kind, base_seed, input_words, total_neurons, interconnect in cases:
            model_path = tmp_path / f"{name}.nir"
            nir.write(str(model_path), graph)
            out_dir = tmp_path / f"{name}_compiled"
            module_name = f"{name}_net"

            rc = run_cli(
                "compile-nir",
                str(model_path),
                "--module-name",
                module_name,
                "--T",
                "512",
                "--source-kind",
                source_kind,
                "--base-seed",
                str(base_seed),
                "-o",
                str(out_dir),
            )

            assert rc == 0
            manifest = json.loads(
                (out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["interconnect"] == interconnect
            assert manifest["total_neurons"] == total_neurons

            sim_dir = tmp_path / f"{name}_network_sim"
            sim_dir.mkdir()
            stdout = _simulate_network_bundle(
                out_dir,
                module_name=module_name,
                input_words=input_words,
                total_neurons=total_neurons,
                tmp_path=sim_dir,
            )
            assert f"network_start {total_neurons}" in stdout
            assert f"network_done {module_name}" in stdout

        output = capsys.readouterr().out
        assert "Interconnect: direct" in output
        assert "Interconnect: aer" in output

    def test_compile_nir_direct_network_matches_fixed_point_reference(self, tmp_path: Path) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "direct_equivalence_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())
        out_dir = tmp_path / "direct_equivalence_compiled"
        module_name = "direct_equivalence_net"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "66",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_direct_equivalence_testbench(module_name),
            tmp_path=tmp_path / "direct_equivalence_sim",
        )
        assert _parse_direct_equivalence_stdout(stdout) == _small_direct_fixed_point_reference(8)

    def test_compile_nir_recurrent_network_matches_fixed_point_reference(
        self, tmp_path: Path
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "recurrent_equivalence_fixture.nir"
        nir.write(str(model_path), _recurrent_lif_nir_graph())
        out_dir = tmp_path / "recurrent_equivalence_compiled"
        module_name = "recurrent_equivalence_net"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "41",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        recurrent_row = next(
            row for row in manifest["sources"] if row["stream_id"] == "conn.lif_to_lif.weight"
        )
        assert recurrent_row["delay_steps"] == 1
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_recurrent_equivalence_testbench(module_name),
            tmp_path=tmp_path / "recurrent_equivalence_sim",
        )
        assert _parse_recurrent_equivalence_stdout(stdout) == _recurrent_fixed_point_reference(8)

    def test_compile_nir_aer_network_matches_fixed_point_reference(self, tmp_path: Path) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "aer_equivalence_fixture.nir"
        nir.write(str(model_path), _aer_lif_nir_graph())
        out_dir = tmp_path / "aer_equivalence_compiled"
        module_name = "aer_equivalence_net"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "11",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["total_neurons"] == 67
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_aer_equivalence_testbench(module_name),
            tmp_path=tmp_path / "aer_equivalence_sim",
        )
        assert _parse_aer_equivalence_stdout(stdout) == _aer_fixed_point_reference(8)

    def test_compile_nir_mixed_network_matches_fixed_point_reference(self, tmp_path: Path) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_equivalence_fixture.nir"
        nir.write(str(model_path), _mixed_li_lif_nir_graph())
        out_dir = tmp_path / "mixed_equivalence_compiled"
        module_name = "mixed_equivalence_net"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "91",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "direct"
        assert manifest["scnir_signal_kinds"] == {"analogue_state": 1, "spike": 1, "weight": 2}
        assert manifest["scnir_signal_routes"]["analogue_state"] == "direct_mac"
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_mixed_equivalence_testbench(module_name),
            tmp_path=tmp_path / "mixed_equivalence_sim",
        )
        assert _parse_mixed_equivalence_stdout(stdout) == _mixed_fixed_point_reference(8)

    def test_compile_nir_can_write_scnir_handoff_audit_report(self, tmp_path: Path) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "audit_handoff_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())
        out_dir = tmp_path / "audit_handoff_compiled"
        module_name = "audit_handoff_net"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "77",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["module_name"] == module_name
        assert report["stream_count"] == 2
        assert report["source_module_count"] == 2

    def test_compile_nir_manifest_and_audit_report_generated_hierarchy(
        self, tmp_path: Path
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "nested_hierarchy_fixture.nir"
        nir.write(str(model_path), _nested_single_port_lif_nir_graph())
        out_dir = tmp_path / "nested_hierarchy_compiled"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "nested_hierarchy_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "77",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        document = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(document)
        assert document["hierarchy"] == [
            {
                "instance_id": "subgraph",
                "module_name": "scnir_subgraph",
                "ports": [
                    {
                        "port_name": "weight_0",
                        "direction": "output",
                        "stream_id": "conn.subgraph__input_to_lif.weight",
                        "signal_kind": "weight",
                        "bit_width": 64,
                    }
                ],
            }
        ]

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_hierarchy_instance_count"] == 1
        assert manifest["scnir_hierarchy_port_count"] == 1
        hierarchy_module = (out_dir / "scnir_subgraph.v").read_text(encoding="utf-8")
        assert "module scnir_subgraph (" in hierarchy_module
        assert "output wire signed [63:0] weight_0" in hierarchy_module
        assert "assign weight_0[0 +: 16] = 16'sh0040;" in hierarchy_module
        assert "assign weight_0[48 +: 16] = 16'sh0020;" in hierarchy_module
        assert "// stream_id: conn.subgraph__input_to_lif.weight" in hierarchy_module

        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["hierarchy_instance_count"] == 1
        assert report["hierarchy_port_count"] == 1
        assert "scnir_subgraph.v" in report["artefacts"]
        assert report["hierarchy_instances"]["subgraph"]["ports"] == [
            {
                "bit_width": 64,
                "direction": "output",
                "port_name": "weight_0",
                "signal_kind": "weight",
                "stream_id": "conn.subgraph__input_to_lif.weight",
            }
        ]

    def test_compile_nir_audits_exact_multiport_multioutput_hierarchy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "nested_multiport_multioutput_fixture.nir"
        model_path.write_bytes(b"synthetic multi-port fixture")
        monkeypatch.setattr(
            nir,
            "read",
            lambda _path: _nested_multiport_multioutput_lif_nir_graph(),
        )
        out_dir = tmp_path / "nested_multiport_multioutput_compiled"

        rc = run_cli(
            "compile-nir",
            str(model_path),
            "--module-name",
            "nested_multiport_multioutput_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "81",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        document = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(document)
        expected_ports = [
            {
                "port_name": "weight_0",
                "direction": "output",
                "stream_id": "conn.subgraph__a_to_lif_a.weight",
                "signal_kind": "weight",
                "bit_width": 16,
            },
            {
                "port_name": "weight_1",
                "direction": "output",
                "stream_id": "conn.subgraph__b_to_lif_b.weight",
                "signal_kind": "weight",
                "bit_width": 16,
            },
        ]
        assert document["hierarchy"] == [
            {
                "instance_id": "subgraph",
                "module_name": "scnir_subgraph",
                "ports": expected_ports,
            }
        ]

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_hierarchy_instance_count"] == 1
        assert manifest["scnir_hierarchy_port_count"] == 2
        assert manifest["scnir_stream_count"] == 4
        assert manifest["scnir_external_inputs"] == [
            {"source": "subgraph__a", "offset": 0, "width": 1},
            {"source": "subgraph__b", "offset": 1, "width": 1},
        ]

        top_module = (out_dir / "nested_multiport_multioutput_net.v").read_text(encoding="utf-8")
        assert "ext_input_0 * scnir_subgraph__weight_0" in top_module
        assert "ext_input_1 * scnir_subgraph__weight_1" in top_module
        assert "scnir_subgraph scnir_subgraph_hierarchy_inst (" in top_module
        assert ".weight_0(scnir_subgraph__weight_0)" in top_module
        assert ".weight_1(scnir_subgraph__weight_1)" in top_module
        hierarchy_module = (out_dir / "scnir_subgraph.v").read_text(encoding="utf-8")
        assert "module scnir_subgraph (" in hierarchy_module
        assert "output wire signed [15:0] weight_0" in hierarchy_module
        assert "output wire signed [15:0] weight_1" in hierarchy_module
        assert "assign weight_0 = 16'sh0080;" in hierarchy_module
        assert "assign weight_1 = 16'shffc0;" in hierarchy_module
        assert "// stream_id: conn.subgraph__a_to_lif_a.weight" in hierarchy_module
        assert "// stream_id: conn.subgraph__b_to_lif_b.weight" in hierarchy_module

        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["hierarchy_instance_count"] == 1
        assert report["hierarchy_port_count"] == 2
        assert report["hierarchy_instances"]["subgraph"]["ports"] == expected_ports
        assert "scnir_subgraph.v" in report["artefacts"]
        assert report["external_input_count"] == 2
        assert report["external_inputs"] == [
            {"source": "subgraph__a", "offset": 0, "width": 1},
            {"source": "subgraph__b", "offset": 1, "width": 1},
        ]

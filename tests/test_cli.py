# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.cli

"""Tests for sc_neurocore.cli."""

import builtins
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import types
from unittest import mock

import numpy as np
import pytest

from sc_neurocore.cli import _cmd_info, _cmd_studio, _format_engine_status, main
from sc_neurocore.ir import SCNIR_SCHEMA_VERSION, validate_scnir_dict


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def _fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _small_lif_nir_graph():
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def _dense_lif_nir_graph():
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([3])}),
            "aff1": nir.Affine(
                weight=np.array(
                    [
                        [0.25, -0.5, 0.125],
                        [0.75, 0.125, -0.375],
                        [-0.25, 0.5, 0.625],
                        [0.0, -0.125, 0.25],
                    ],
                    dtype=np.float32,
                ),
                bias=np.zeros(4, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "aff2": nir.Affine(
                weight=np.array(
                    [
                        [0.5, -0.25, 0.125, 0.0],
                        [-0.125, 0.375, -0.5, 0.25],
                    ],
                    dtype=np.float32,
                ),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _aer_lif_nir_graph():
    nir = pytest.importorskip("nir")
    n_in = 4
    n_hidden = 65
    n_out = 2
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=np.full((n_hidden, n_in), 0.125, dtype=np.float32),
                bias=np.zeros(n_hidden, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(n_hidden, 20.0),
                r=np.ones(n_hidden),
                v_leak=np.zeros(n_hidden),
                v_threshold=np.ones(n_hidden),
            ),
            "aff2": nir.Affine(
                weight=np.full((n_out, n_hidden), -0.0625, dtype=np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _mixed_li_lif_nir_graph():
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _sobol_source_smoke_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [15:0] threshold = 16'h9000;
    wire bit_out;
    wire [15:0] value;
    wire [15:0] index;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .threshold(threshold),
        .bit_out(bit_out),
        .value(value),
        .index(index)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        #1 $display("sample0 %04h %0d %0d", value, bit_out, index);
        $finish;
    end
endmodule
"""


def _simulate_single_source_module(module_name: str, verilog: str, tmp_path) -> str:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None, "Icarus Verilog must be installed"

    rtl_path = tmp_path / f"{module_name}.v"
    tb_path = tmp_path / "tb_source.v"
    out_path = tmp_path / "tb_source.out"
    rtl_path.write_text(verilog, encoding="utf-8")
    tb_path.write_text(_sobol_source_smoke_testbench(module_name), encoding="utf-8")

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [vvp, str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
    return run_result.stdout


def test_version_flag(capsys):
    rc = _run_main("--version")
    assert rc == 0
    from sc_neurocore import __version__

    assert __version__ in capsys.readouterr().out


def test_info_command(capsys):
    fake_jax = _fake_module("jax", __version__="0.0-test")
    with mock.patch.dict("sys.modules", {"jax": fake_jax}):
        rc = _run_main("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "sc-neurocore" in out
    assert "Python" in out
    assert "NumPy" in out
    assert "JAX: 0.0-test" in out


def test_no_command_prints_help(capsys):
    rc = _run_main()
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_info_without_rust_engine(capsys):
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": None}):
        rc = _cmd_info()
    assert rc == 0
    assert "not available" in capsys.readouterr().out


def test_info_reports_engine_version_mismatch(capsys):
    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="0.0.0",
        simd_tier=lambda: "mock-tier",
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        rc = _cmd_info()
    assert rc == 0
    out = capsys.readouterr().out
    assert "version mismatch" in out
    assert "expected" in out


def test_info_uses_metadata_without_importing_optional_jax(capsys):
    def fake_version(name: str) -> str:
        if name == "jax":
            return "0.0-meta"
        if name == "numpy":
            return "0.0-numpy"
        raise importlib.metadata.PackageNotFoundError(name)

    with mock.patch("sc_neurocore.cli.importlib.metadata.version", side_effect=fake_version):
        rc = _cmd_info()
    assert rc == 0
    out = capsys.readouterr().out
    assert "JAX: 0.0-meta" in out


def test_info_ignores_missing_optional_metadata(capsys):
    with (
        mock.patch.dict("sys.modules", {"numpy": None, "jax": None}),
        mock.patch(
            "sc_neurocore.cli.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("missing"),
        ),
    ):
        rc = _cmd_info()
    assert rc == 0
    assert "NumPy:" not in capsys.readouterr().out


def test_format_engine_status_without_simd_tier():
    fake = _fake_module("sc_neurocore_engine", __version__="3.13.0")
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


def test_format_engine_status_with_broken_simd_tier():
    def explode():
        raise RuntimeError("no simd")

    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="3.13.0",
        simd_tier=explode,
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


def test_benchmark_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("benchmark")
    assert rc == 0
    m.assert_called_once()


def test_preflight_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("preflight")
    assert rc == 0
    m.assert_called_once()


@pytest.mark.skipif(
    not importlib.util.find_spec("uvicorn"),
    reason="uvicorn not installed (studio extra)",
)
def test_studio_launches_uvicorn(capsys):
    with (
        mock.patch("uvicorn.run") as m_uvicorn,
        mock.patch("webbrowser.open") as m_browser,
    ):
        rc = _cmd_studio(port=8001)
    assert rc == 0
    m_uvicorn.assert_called_once()
    m_browser.assert_called_once_with("http://127.0.0.1:8001")


def test_studio_missing_fastapi(capsys):
    real_import = builtins.__import__

    def block_uvicorn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=block_uvicorn):
        rc = _cmd_studio(port=8001)
    assert rc == 1
    assert "pip install" in capsys.readouterr().out


def test_studio_command_via_main(capsys):
    with (
        mock.patch("sc_neurocore.cli._cmd_studio", return_value=0) as m_studio,
    ):
        rc = _run_main("studio")
    assert rc == 0
    m_studio.assert_called_once_with(8001)


# ---------------------------------------------------------------------------
# Deploy command
# ---------------------------------------------------------------------------


class TestDeployCommand:
    """Tests for `sc-neurocore deploy ...` and the underlying _cmd_deploy."""

    def test_deploy_without_model_arg_returns_1(self, capsys):
        """`sc-neurocore deploy` with no model argument prints usage and exits 1."""
        rc = _run_main("deploy")
        assert rc == 1
        out = capsys.readouterr().out
        assert "deploy requires a model file" in out

    def test_deploy_unsupported_extension_returns_1(self, capsys, tmp_path):
        """A model with an unsupported extension exits 1 with a clear message."""
        from sc_neurocore.cli import _cmd_deploy

        bogus = tmp_path / "model.onnx"
        bogus.write_bytes(b"\x00")
        rc = _cmd_deploy(str(bogus), "ice40", str(tmp_path / "out"), dt=1.0, bitstream_length=256)
        assert rc == 1
        assert "unsupported file format" in capsys.readouterr().out

    def test_deploy_pytorch_writes_verilog_and_hdl_dir(self, tmp_path, capsys):
        """A `.pt` checkpoint with two Linear layers compiles to a Verilog project."""
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        # Build a minimal 2-layer Linear stack and save its state_dict
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)

        out_dir = tmp_path / "deploy_out"
        rc = _cmd_deploy(str(ckpt), "ice40", str(out_dir), dt=1.0, bitstream_length=64)
        assert rc == 0

        # Generated SystemVerilog
        sv = out_dir / "sc_deploy_lif.sv"
        assert sv.exists() and sv.stat().st_size > 0

        # Makefile (Yosys flow → ice40)
        assert (out_dir / "Makefile").exists()

        # README in the deploy dir
        readme = (out_dir / "README.md").read_text()
        assert "ice40" in readme
        power_model = json.loads((out_dir / "power_thermal_model.json").read_text())
        assert power_model["source_mode"] == "pre_silicon_estimate"
        assert power_model["workload"]["layer_sizes"] == [[4, 8], [8, 2]]

    def test_deploy_emits_vivado_tcl_for_artix7(self, tmp_path):
        """artix7 target should emit a project.tcl, not a Makefile."""
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)

        out_dir = tmp_path / "vivado_out"
        rc = _cmd_deploy(str(ckpt), "artix7", str(out_dir), dt=1.0, bitstream_length=64)
        assert rc == 0
        assert (out_dir / "project.tcl").exists()
        assert not (out_dir / "Makefile").exists()
        # README mentions artix7
        assert "artix7" in (out_dir / "README.md").read_text()

    def test_deploy_via_main_dispatcher(self, tmp_path):
        """`sc-neurocore deploy model.pt --target ice40 -o ...` end-to-end via main()."""
        torch = pytest.importorskip("torch")

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ckpt = tmp_path / "m.pt"
        torch.save(model.state_dict(), ckpt)

        out = tmp_path / "deployed"
        rc = _run_main("deploy", str(ckpt), "--target", "ice40", "-o", str(out))
        assert rc == 0
        assert (out / "sc_deploy_lif.sv").exists()


# ---------------------------------------------------------------------------
# NIR silicon mapping command
# ---------------------------------------------------------------------------


class TestMapNirCommand:
    """Tests for `sc-neurocore map-nir ...`."""

    def test_map_nir_rejects_non_nir_extension(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_map_nir

        rc = _cmd_map_nir(
            str(tmp_path / "model.pt"),
            str(tmp_path / "out"),
            "loihi2",
            dt=1.0,
            bitstream_length=256,
        )

        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_map_nir_writes_report_with_mocked_nir_import(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_map_nir

        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")
        fake_graph = mock.MagicMock()
        fake_network = types.SimpleNamespace(
            nodes={
                "input": {"node_type": "Input", "shape": (2,)},
                "dense": {"node_type": "Linear", "weight": (3, 2)},
                "output": {"node_type": "Output", "shape": (3,)},
            },
            topo_order=["input", "dense", "output"],
            edges=[("input", "dense"), ("dense", "output")],
        )
        fake_nir = _fake_module("nir", read=mock.MagicMock(return_value=fake_graph))

        with (
            mock.patch.dict("sys.modules", {"nir": fake_nir}),
            mock.patch("sc_neurocore.nir_bridge.from_nir", return_value=fake_network),
        ):
            rc = _cmd_map_nir(
                str(nir_path),
                str(tmp_path / "mapping"),
                "loihi2,spinnaker2,akida",
                dt=0.5,
                bitstream_length=256,
            )

        report = json.loads(
            (tmp_path / "mapping" / "nir_silicon_mapping_report.json").read_text(encoding="utf-8")
        )
        assert rc == 0
        assert [target["target_id"] for target in report["targets"]] == [
            "loihi2",
            "spinnaker2",
            "akida",
        ]
        assert report["targets"][0]["summary"]["estimated_synapses"] == 6
        assert "NIR silicon mapping report generated" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NIR FPGA compilation command
# ---------------------------------------------------------------------------


class TestCompileNirCommand:
    """Tests for `sc-neurocore compile-nir ...` exported artefacts."""

    def test_compile_nir_writes_scnir_source_bundle_and_simulates_source(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())

        out_dir = tmp_path / "compiled"
        rc = _run_main(
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
        assert manifest["schema_version"] == "sc-neurocore.scnir.hdl-sources.v0.1"
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

    def test_compile_nir_writes_valid_dense_scnir_document(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "dense_fixture.nir"
        nir.write(str(model_path), _dense_lif_nir_graph())

        out_dir = tmp_path / "dense_compiled"
        rc = _run_main(
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

    def test_compile_nir_records_aer_interconnect_in_manifest(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "aer_fixture.nir"
        nir.write(str(model_path), _aer_lif_nir_graph())

        out_dir = tmp_path / "aer_compiled"
        rc = _run_main(
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

    def test_compile_nir_records_mixed_signal_summary_in_manifest(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_signal_fixture.nir"
        nir.write(str(model_path), _mixed_li_lif_nir_graph())

        out_dir = tmp_path / "mixed_signal_compiled"
        rc = _run_main(
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


# ---------------------------------------------------------------------------
# Self-hosted hub command
# ---------------------------------------------------------------------------


class TestHubInitCommand:
    """Tests for `sc-neurocore hub-init ...`."""

    def test_hub_init_writes_bundle(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_hub_init

        out = tmp_path / "hub"
        rc = _cmd_hub_init(
            str(out),
            port=8111,
            bind_host="10.0.0.5",
            image="sc-neurocore:test",
            offline=False,
        )

        assert rc == 0
        assert (out / "docker-compose.yml").exists()
        manifest = json.loads((out / "hub_manifest.json").read_text(encoding="utf-8"))
        assert manifest["services"]["studio"]["url"] == "http://10.0.0.5:8111"
        assert manifest["network_policy"]["ingress_scope"] == "private_network"
        assert manifest["network_policy"]["offline_environment"]["SC_NEUROCORE_HUB_OFFLINE"] == "0"
        assert "image: sc-neurocore:test" in (out / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        assert "hub bundle generated" in capsys.readouterr().out

    def test_hub_init_rejects_invalid_port(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_hub_init

        rc = _cmd_hub_init(str(tmp_path / "hub"), port=0)

        assert rc == 1
        assert "studio_port must be in the range" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for `sc-neurocore serve ...` and the underlying _cmd_serve."""

    def test_serve_without_model_arg_returns_1(self, capsys):
        """`sc-neurocore serve` with no model argument prints usage and exits 1."""
        rc = _run_main("serve")
        assert rc == 1
        out = capsys.readouterr().out
        assert "serve requires a model file" in out

    def test_serve_rejects_non_nir_extension(self, capsys):
        """`.pt` and other extensions are not yet supported by serve; exit 1."""
        from sc_neurocore.cli import _cmd_serve

        rc = _cmd_serve("model.pt", port=8001, dt=1.0)
        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_serve_loads_nir_and_blocks_in_server(self, tmp_path, capsys):
        """Successful path: read NIR, build Network, start blocking SpikeServer."""
        from sc_neurocore.cli import _cmd_serve

        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")  # contents are mocked away

        # Fake graph and Network — we only need topo_order length
        fake_graph = mock.MagicMock()
        fake_network = mock.MagicMock()
        fake_network.topo_order = ["a", "b", "c"]

        # Build a fake SpikeServer that records start() and returns immediately.
        fake_server_instance = mock.MagicMock()
        fake_server_cls = mock.MagicMock(return_value=fake_server_instance)

        # Patch the lazy imports inside _cmd_serve via sys.modules
        fake_nir = _fake_module("nir", read=mock.MagicMock(return_value=fake_graph))
        fake_bridge = _fake_module(
            "sc_neurocore.nir_bridge",
            from_nir=mock.MagicMock(return_value=fake_network),
        )
        fake_serve_mod = _fake_module(
            "sc_neurocore.serve",
            SpikeServer=fake_server_cls,
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "nir": fake_nir,
                "sc_neurocore.nir_bridge": fake_bridge,
                "sc_neurocore.serve": fake_serve_mod,
            },
        ):
            rc = _cmd_serve(str(nir_path), port=8123, dt=1.0)

        assert rc == 0
        # SpikeServer was constructed with the fake network and the given port
        fake_server_cls.assert_called_once_with(fake_network, port=8123)
        # And started in blocking mode
        fake_server_instance.start.assert_called_once_with(blocking=True)
        # Confirmation print mentions the node count
        assert "Loaded NIR graph with 3 nodes" in capsys.readouterr().out

    def test_serve_via_main_dispatcher_routes_to_cmd_serve(self, tmp_path):
        """`sc-neurocore serve model.nir --port N` reaches _cmd_serve with the right args."""
        nir_path = tmp_path / "x.nir"
        nir_path.write_bytes(b"")
        with mock.patch("sc_neurocore.cli._cmd_serve", return_value=0) as m_serve:
            rc = _run_main("serve", str(nir_path), "--port", "9000", "--dt", "1.0")
        assert rc == 0
        m_serve.assert_called_once_with(str(nir_path), 9000, 1.0)

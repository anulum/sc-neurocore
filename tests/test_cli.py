# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.cli

"""Tests for sc_neurocore.cli."""

import builtins
import importlib.util
import types
from unittest import mock

import pytest

from sc_neurocore.cli import _cmd_info, _cmd_studio, _format_engine_status, main


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def _fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


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


def test_info_ignores_broken_optional_jax_import(capsys):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax":
            raise AttributeError("broken jax")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        rc = _cmd_info()
    assert rc == 0
    assert "JAX:" not in capsys.readouterr().out


def test_info_ignores_broken_optional_numpy_import(capsys):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise RuntimeError("broken numpy")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
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

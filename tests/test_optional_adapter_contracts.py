# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional Adapter Contract Tests

"""Contract for optional adapter control paths that run without external services."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
import builtins

import numpy as np
import pytest

from sc_neurocore.accel import mojo_dispatch
from sc_neurocore.debug import hil_server
from sc_neurocore.formal.lean_bridge import FormalProofEngine


def test_mojo_dispatch_numpy_backend_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mojo_dispatch, "_MOJO_AVAILABLE", False)
    monkeypatch.setattr(mojo_dispatch, "_MOJO_BIN", None)
    monkeypatch.setattr(mojo_dispatch, "_RUST_AVAILABLE", False)

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 8, dtype=np.uint8)
    packed = mojo_dispatch.pack_bitstream(bits)
    assert packed.dtype == np.uint64
    assert mojo_dispatch.popcount(packed) == int(bits.sum())
    assert mojo_dispatch.scc(packed, packed, bit_length=len(bits)) == pytest.approx(1.0)
    assert mojo_dispatch.scc(packed, packed, bit_length=0) == 0.0
    assert (
        mojo_dispatch.scc(
            packed, zeros := mojo_dispatch.pack_bitstream(np.zeros_like(bits)), bit_length=len(bits)
        )
        == 0.0
    )

    ones = mojo_dispatch.pack_bitstream(np.ones_like(bits))
    np.testing.assert_array_equal(mojo_dispatch.sc_and(packed, ones), packed)
    np.testing.assert_array_equal(mojo_dispatch.sc_or(packed, zeros), packed)
    np.testing.assert_array_equal(mojo_dispatch.sc_xor(packed, zeros), packed)
    np.testing.assert_array_equal(mojo_dispatch.sc_mux(packed, zeros, ones), packed)

    weights = np.vstack([packed, zeros])
    mac = mojo_dispatch.vec_mac(weights, packed)
    np.testing.assert_array_equal(mac, np.array([int(bits.sum()), 0], dtype=np.int64))

    info = mojo_dispatch.backend_info()
    assert info["active_backend"] == "numpy"
    assert info["mojo_available"] is False
    assert info["rust_ffi_available"] is False


def test_mojo_dispatch_internal_fallback_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([0xFFFF_FFFF, 0], dtype=np.uint32)

    assert mojo_dispatch._py_popcount32(0xFFFF_FFFF) == 32
    assert mojo_dispatch._py_popcount_array(arr) == 32

    original_import = builtins.__import__

    def blocked_import(name: str, *args, **kwargs):
        if name == "sc_neurocore._native":
            raise ImportError("native bridge unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert mojo_dispatch._detect_rust() is False


def test_mojo_dispatch_negative_correlation_path() -> None:
    a = mojo_dispatch.pack_bitstream(np.array([1, 1, 0, 0] * 16, dtype=np.uint8))
    b = mojo_dispatch.pack_bitstream(np.array([0, 0, 1, 1] * 16, dtype=np.uint8))

    assert mojo_dispatch.scc(a, b, bit_length=64) == pytest.approx(-1.0)


def test_mojo_detection_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mojo_dispatch.os.path, "expanduser", lambda _: str(tmp_path / "pixi"))
    monkeypatch.setattr(mojo_dispatch.os.path, "dirname", lambda _: str(tmp_path / "pkg"))
    monkeypatch.setattr(mojo_dispatch.os.path, "normpath", lambda value: value)
    monkeypatch.setattr(mojo_dispatch.os.path, "exists", lambda _: False)
    assert mojo_dispatch._detect_mojo() is False

    def exists(path: str) -> bool:
        return path == str(tmp_path / "pixi") or path.endswith("kernels.mojo")

    monkeypatch.setattr(mojo_dispatch.os.path, "exists", exists)
    assert mojo_dispatch._detect_mojo() is True
    assert mojo_dispatch._MOJO_BIN is not None


def test_hil_daemon_reports_missing_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hil_server.sysconfig, "get_path", lambda _: str(tmp_path / "scripts"))

    with pytest.raises(FileNotFoundError, match="HIL Debugger"):
        hil_server.HILServerDaemon(_go_dir=tmp_path / "missing")


def test_hil_daemon_uses_installed_binary_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    installed = scripts / "hil_debugger"
    installed.write_text("#!/bin/sh\n")
    monkeypatch.setattr(hil_server.sysconfig, "get_path", lambda _: str(scripts))

    daemon = hil_server.HILServerDaemon(_go_dir=tmp_path / "missing")

    assert daemon._go_dir == scripts


def test_hil_daemon_start_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    go_dir = tmp_path / "hil"
    go_dir.mkdir()
    daemon = hil_server.HILServerDaemon(_go_dir=go_dir)

    class RunningProcess:
        stderr = None

        def poll(self) -> None:
            return None

    daemon._process = RunningProcess()  # type: ignore[assignment]
    assert daemon.start() is True

    daemon._process = None
    monkeypatch.setattr(
        hil_server.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0], stderr=b"compile failed")
        ),
    )
    assert daemon.start(build=True) is False

    monkeypatch.setattr(hil_server.subprocess, "run", lambda *args, **kwargs: None)
    assert daemon.start(build=False) is False


def test_hil_daemon_start_success_invokes_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    go_dir = tmp_path / "hil"
    go_dir.mkdir()
    binary = go_dir / "hil_debugger"
    binary.write_text("#!/bin/sh\n")
    daemon = hil_server.HILServerDaemon(port=8123, _go_dir=go_dir)

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def poll(self) -> None:
            return None

    monkeypatch.setattr(hil_server.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(hil_server.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(daemon, "_wait_for_ready", lambda: True)

    assert daemon.start(build=True) is True
    assert isinstance(daemon._process, FakePopen)
    assert daemon._process.kwargs["env"]["HIL_PORT"] == "8123"  # type: ignore[union-attr]


def test_hil_wait_ready_and_stop_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    go_dir = tmp_path / "hil"
    go_dir.mkdir()
    daemon = hil_server.HILServerDaemon(port=9001, _go_dir=go_dir)

    class CrashedProcess:
        stderr = SimpleNamespace(read=lambda: b"boom")

        def poll(self) -> int:
            return 1

    daemon._process = CrashedProcess()  # type: ignore[assignment]
    assert daemon._wait_for_ready(timeout_sec=1) is False

    class KillableProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> None:
            raise subprocess.TimeoutExpired("hil_debugger", timeout)

        def kill(self) -> None:
            self.killed = True

    process = KillableProcess()
    daemon._process = process  # type: ignore[assignment]
    assert daemon.is_running is True
    daemon.stop()
    assert process.terminated is True
    assert process.killed is True
    assert daemon._process is None


def test_hil_wait_ready_success_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    go_dir = tmp_path / "hil"
    go_dir.mkdir()
    daemon = hil_server.HILServerDaemon(port=9002, _go_dir=go_dir)

    class RunningProcess:
        stderr = None

        def poll(self) -> None:
            return None

    class Response:
        status = 200

    class HealthyConnection:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def request(self, method: str, path: str) -> None:
            assert (method, path) == ("GET", "/health")

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    daemon._process = RunningProcess()  # type: ignore[assignment]
    monkeypatch.setattr(hil_server.http.client, "HTTPConnection", HealthyConnection)
    assert daemon._wait_for_ready(timeout_sec=1) is True

    class RefusingConnection:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def request(self, method: str, path: str) -> None:
            raise ConnectionError("not ready")

        def close(self) -> None:
            pass

    stopped = {"called": False}
    daemon._process = RunningProcess()  # type: ignore[assignment]
    ticks = iter([0.0, 0.5, 2.0])
    monkeypatch.setattr(hil_server.http.client, "HTTPConnection", RefusingConnection)
    monkeypatch.setattr(hil_server.time, "time", lambda: next(ticks))
    monkeypatch.setattr(hil_server.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon, "stop", lambda: stopped.__setitem__("called", True))
    assert daemon._wait_for_ready(timeout_sec=1) is False
    assert stopped["called"] is True


def test_formal_proof_engine_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.shutil.which", lambda _: None)
    engine = FormalProofEngine()
    engine.proof_file = tmp_path / "missing.lean"

    assert engine.is_available() is False
    assert engine.list_axioms() == []
    assert engine.list_theorems() == []
    assert engine.proof_inventory_matches() is False
    assert engine.check_proofs() is False


def test_formal_proof_engine_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proof_file = tmp_path / "safety_bounds.lean"
    proof_file.write_text(
        "\n".join(
            [
                "axiom sc_precision_numerator_bound : True",
                "axiom sc_add_preserves_range : True",
                "theorem ok : True := by trivial",
                "  axiom nested_text_is_not_a_declaration : True",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.shutil.which", lambda _: "/usr/bin/lean")
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.EXPECTED_THEOREMS", ("ok",))
    engine = FormalProofEngine()
    engine.proof_file = proof_file

    assert engine.list_axioms() == ["sc_precision_numerator_bound", "sc_add_preserves_range"]
    assert engine.list_theorems() == ["ok"]
    assert engine.axiom_inventory_matches() is True
    assert engine.theorem_inventory_matches() is True
    assert engine.proof_inventory_matches() is True

    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""),
    )
    assert engine.is_available() is True
    assert engine.check_proofs() is True

    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="error: bad theorem", stderr=""),
    )
    assert engine.check_proofs() is False

    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0], stderr="native failure")
        ),
    )
    assert engine.check_proofs() is False


def test_formal_proof_engine_rejects_unexpected_axiom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proof_file = tmp_path / "safety_bounds.lean"
    proof_file.write_text(
        "\n".join(
            [
                "axiom sc_precision_numerator_bound : True",
                "axiom sc_add_preserves_range : True",
                "axiom unreviewed_shortcut : True",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.shutil.which", lambda _: "/usr/bin/lean")
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.EXPECTED_THEOREMS", ())
    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""),
    )

    engine = FormalProofEngine()
    engine.proof_file = proof_file

    assert engine.axiom_inventory_matches() is False
    assert engine.check_proofs() is False


def test_formal_proof_engine_rejects_theorem_inventory_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proof_file = tmp_path / "safety_bounds.lean"
    proof_file.write_text(
        "\n".join(
            [
                "axiom sc_precision_numerator_bound : True",
                "axiom sc_add_preserves_range : True",
                "theorem retained_contract : True := by trivial",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.shutil.which", lambda _: "/usr/bin/lean")
    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.EXPECTED_THEOREMS",
        ("retained_contract", "missing_contract"),
    )
    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""),
    )

    engine = FormalProofEngine()
    engine.proof_file = proof_file

    assert engine.axiom_inventory_matches() is True
    assert engine.theorem_inventory_matches() is False
    assert engine.check_proofs() is False


def test_formal_proof_engine_times_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proof_file = tmp_path / "safety_bounds.lean"
    proof_file.write_text(
        "\n".join(
            [
                "axiom sc_precision_numerator_bound : True",
                "axiom sc_add_preserves_range : True",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sc_neurocore.formal.lean_bridge.shutil.which", lambda _: "/usr/bin/lean")
    monkeypatch.setattr(
        "sc_neurocore.formal.lean_bridge.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], timeout=300)
        ),
    )

    engine = FormalProofEngine()
    engine.proof_file = proof_file

    assert engine.check_proofs() is False

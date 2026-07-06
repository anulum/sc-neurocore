# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NeuroCore PYNQ FPGA driver

# tests/test_pynq_driver.py
"""Tests for SC-NeuroCore PYNQ FPGA driver."""

import pytest
import numpy as np
import sys
import types
import sc_neurocore.drivers.sc_neurocore_driver as pynq_driver
from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge
from sc_neurocore.drivers.sc_neurocore_driver import SC_NeuroCore_Driver, RealityHardwareError


def test_driver_emulation_mode():
    """Verify driver works in emulation mode."""
    driver = SC_NeuroCore_Driver(mode="EMULATION")

    assert driver.mode == "EMULATION"
    assert driver.overlay is None

    print("Driver emulation mode verified.")


def test_driver_write_layer_params():
    """Verify driver can write layer parameters in emulation mode."""
    driver = SC_NeuroCore_Driver(mode="EMULATION")

    # Should not raise in emulation mode
    driver.write_layer_params(layer_id=1, params={"gain": 0.5, "threshold": 1.0})
    driver.write_layer_params(layer_id=2, params={"gain": 2.0})

    print("Driver write_layer_params verified.")


def test_driver_write_layer_params_hardware_q16_16_encoding():
    """HARDWARE mode writes gain/threshold as Q16.16 AXI-Lite registers."""

    class FakeLayerIP:
        def __init__(self):
            self.writes: list[tuple[int, int]] = []

        def write(self, offset: int, value: int) -> None:
            self.writes.append((offset, value))

    class FakeOverlay:
        def __init__(self):
            self.scpn_layer_3_0 = FakeLayerIP()

    driver = SC_NeuroCore_Driver(mode="EMULATION")
    driver.mode = "HARDWARE"
    driver.overlay = FakeOverlay()

    driver.write_layer_params(layer_id=3, params={"gain": 0.5, "threshold": 1.25})

    assert driver.overlay.scpn_layer_3_0.writes == [
        (0x10, 32768),
        (0x14, 81920),
    ]


def test_driver_write_layer_params_hardware_rejects_missing_layer():
    """HARDWARE mode fails closed when the target layer IP is absent."""
    driver = SC_NeuroCore_Driver(mode="EMULATION")
    driver.mode = "HARDWARE"
    driver.overlay = object()

    with pytest.raises(ValueError, match="Layer 9 not found in hardware"):
        driver.write_layer_params(layer_id=9, params={"gain": 1.0})


def test_driver_run_step():
    """Verify run_step returns output in emulation mode."""
    driver = SC_NeuroCore_Driver(mode="EMULATION")

    input_vector = np.random.rand(16)
    output = driver.run_step(input_vector)

    assert isinstance(output, np.ndarray)
    assert output.shape == (16,)
    assert np.all(np.isfinite(output))

    print("Driver run_step verified.")


def test_driver_hardware_mode_fails_without_fpga():
    """Verify hardware mode fails gracefully without FPGA."""
    with pytest.raises(RealityHardwareError):
        driver = SC_NeuroCore_Driver(mode="HARDWARE")


def test_driver_hardware_mode_uses_install_fallback_bitstream(monkeypatch):
    """HARDWARE mode resolves the installed overlay path when local bitstream is absent."""
    loaded_paths: list[str] = []

    class FakeOverlay:
        def __init__(self, path: str):
            loaded_paths.append(path)
            self.scpn_layer_1_0 = object()

    fake_pynq = types.ModuleType("pynq")
    fake_pynq.Overlay = FakeOverlay
    fake_pynq.allocate = object()
    monkeypatch.setitem(sys.modules, "pynq", fake_pynq)

    fallback = "/usr/local/lib/pynq/overlays/sc_neurocore/installed.bit"
    monkeypatch.setattr(
        pynq_driver.os.path,
        "exists",
        lambda path: path == fallback,
    )

    driver = SC_NeuroCore_Driver(bitstream_path="installed.bit", mode="HARDWARE")

    assert driver.bitstream_path == fallback
    assert loaded_paths == [fallback]


def test_driver_hardware_mode_rejects_overlay_without_expected_ip(monkeypatch):
    """Loaded overlays must expose the expected SCPN layer IP."""

    class FakeOverlay:
        def __init__(self, path: str):
            self.path = path

    fake_pynq = types.ModuleType("pynq")
    fake_pynq.Overlay = FakeOverlay
    fake_pynq.allocate = object()
    monkeypatch.setitem(sys.modules, "pynq", fake_pynq)
    monkeypatch.setattr(pynq_driver.os.path, "exists", lambda _path: True)

    with pytest.raises(RealityHardwareError, match="does not contain SCPN Layer 1 IP"):
        SC_NeuroCore_Driver(bitstream_path="wrong.bit", mode="HARDWARE")


def test_driver_hardware_mode_wraps_overlay_runtime_errors(monkeypatch):
    """Overlay loader runtime failures are reported as RealityHardwareError."""

    class FailingOverlay:
        def __init__(self, path: str):
            raise RuntimeError(f"cannot load {path}")

    fake_pynq = types.ModuleType("pynq")
    fake_pynq.Overlay = FailingOverlay
    fake_pynq.allocate = object()
    monkeypatch.setitem(sys.modules, "pynq", fake_pynq)
    monkeypatch.setattr(pynq_driver.os.path, "exists", lambda _path: True)

    with pytest.raises(RealityHardwareError, match="cannot load broken.bit"):
        SC_NeuroCore_Driver(bitstream_path="broken.bit", mode="HARDWARE")


def test_driver_hardware_mode_wraps_missing_bitstream(monkeypatch):
    """A bitstream absent both locally and in the install path fails closed."""
    fake_pynq = types.ModuleType("pynq")
    fake_pynq.Overlay = object
    fake_pynq.allocate = object
    monkeypatch.setitem(sys.modules, "pynq", fake_pynq)
    # Neither the local nor the /usr/local install path exists.
    monkeypatch.setattr(pynq_driver.os.path, "exists", lambda _path: False)

    with pytest.raises(RealityHardwareError, match="Hardware initialization failed"):
        SC_NeuroCore_Driver(bitstream_path="missing.bit", mode="HARDWARE")


def test_driver_invalid_mode():
    """Verify invalid mode raises error."""
    with pytest.raises(ValueError):
        driver = SC_NeuroCore_Driver(mode="INVALID")


# ---------------------------------------------------------------------------
# EMULATION run_step determinism (task #29 fix verification)
# ---------------------------------------------------------------------------


class TestRunStepDeterminism:
    """Two drivers built with the same seed produce identical run_step output."""

    def test_run_step_same_seed_identical_first_call(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=123)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=123)
        np.testing.assert_array_equal(a.run_step(None), b.run_step(None))

    def test_run_step_same_seed_identical_sequence(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=99)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=99)
        for _ in range(50):
            np.testing.assert_array_equal(a.run_step(None), b.run_step(None))

    def test_run_step_different_seeds_differ(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=1)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=2)
        out_a = a.run_step(None)
        out_b = b.run_step(None)
        # Two distinct seeds: shape matches, values differ
        assert out_a.shape == out_b.shape == (16,)
        assert not np.array_equal(out_a, out_b)

    def test_run_step_global_numpy_seed_does_not_leak(self):
        """np.random.seed(...) between constructions must not affect output."""
        np.random.seed(0)
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        out_a = a.run_step(None)

        np.random.seed(99999)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        out_b = b.run_step(None)
        np.testing.assert_array_equal(out_a, out_b)

    def test_run_step_default_seed_is_42(self):
        a = SC_NeuroCore_Driver(mode="EMULATION")
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        np.testing.assert_array_equal(a.run_step(None), b.run_step(None))


class TestVerifyHardwareLink:
    """verify_link CLI smoke tests (closes task #31)."""

    def test_extras_false_fpga_only(self, capsys):
        """extras=False skips Evo 2 + Opentrons probes."""
        from sc_neurocore.drivers.verify_hardware_link import verify_link

        verify_link(extras=False)
        out = capsys.readouterr().out
        assert "[1/1]" in out
        assert "FPGA only" in out
        # Evo 2 + Opentrons headers must be absent
        assert "[2/" not in out
        assert "[3/" not in out
        assert "Genomic" not in out
        assert "Robotics" not in out

    def test_extras_true_runs_all_three_probes(self, capsys):
        """extras=True (default) runs all three probes including the
        sibling-repo imports.

        On environments where the sibling modules are absent (the
        common case outside the GOTM monorepo), the probes report
        "FAILURE: <module> not on PYTHONPATH" cleanly without
        manipulating sys.path.
        """
        from sc_neurocore.drivers.verify_hardware_link import verify_link

        verify_link(extras=True)
        out = capsys.readouterr().out
        assert "[1/3]" in out
        assert "[2/3]" in out
        assert "[3/3]" in out
        assert "Genomic" in out
        assert "Robotics" in out

    def test_extras_default_is_true(self, capsys):
        from sc_neurocore.drivers.verify_hardware_link import verify_link

        verify_link()  # default
        out = capsys.readouterr().out
        assert "[3/3]" in out

    def test_no_sys_path_mutation(self):
        """verify_link must not mutate sys.path (closes the cross-repo bug)."""
        import sys

        from sc_neurocore.drivers.verify_hardware_link import verify_link

        before = list(sys.path)
        verify_link(extras=True)
        after = list(sys.path)
        assert before == after, "verify_link mutated sys.path"

    def test_fpga_probe_reports_success_when_driver_connects(self, monkeypatch, capsys):
        """A driver that constructs cleanly drives the SUCCESS branch."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        monkeypatch.setattr(vhl, "SC_NeuroCore_Driver", lambda mode: object())
        vhl.verify_link(extras=False)
        out = capsys.readouterr().out
        assert "SUCCESS: PYNQ-Z2 Detected" in out

    def test_fpga_probe_reports_unexpected_runtime_error(self, monkeypatch, capsys):
        """A raw OSError/RuntimeError (not RealityHardwareError) hits the ERROR branch."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        def boom(mode):
            raise RuntimeError("bus fault")

        monkeypatch.setattr(vhl, "SC_NeuroCore_Driver", boom)
        vhl.verify_link(extras=False)
        out = capsys.readouterr().out
        assert "ERROR: Unexpected failure: bus fault" in out

    def test_genomic_probe_handles_present_but_unreachable_evo2(self, monkeypatch, capsys):
        """When Evo 2 is importable but its server is down, the probe warns cleanly."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        evo_mod = types.ModuleType("scpn_evo2_real_interface")

        class Evo2RealInterface:
            def connect(self):
                raise ConnectionError("server down")

        evo_mod.Evo2RealInterface = Evo2RealInterface
        monkeypatch.setitem(sys.modules, "scpn_evo2_real_interface", evo_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Evo 2 Server unreachable" in out

    def test_robotics_probe_reports_opentrons_online(self, monkeypatch, capsys):
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self):
                return True

        ot_mod.OpentronsVerifier = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Opentrons OT-2 Online" in out

    def test_robotics_probe_reports_opentrons_offline(self, monkeypatch, capsys):
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self):
                return False

        ot_mod.OpentronsVerifier = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Robot offline" in out

    def test_robotics_probe_handles_opentrons_error(self, monkeypatch, capsys):
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self):
                raise RuntimeError("robot fault")

        ot_mod.OpentronsVerifier = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "robot fault" in out


class TestPhysicalTwinBridge:
    """PhysicalTwinBridge exposes honest emulation and explicit TCP modes."""

    def test_default_bridge_is_emulation_without_stdout(self, capsys):
        bridge = PhysicalTwinBridge(seed=7)

        assert bridge.mode == "EMULATION"
        assert bridge.connected is True
        assert not hasattr(bridge, "_TODO" + "_HIL")
        assert capsys.readouterr().out == ""

    def test_emulation_mode_is_deterministic_and_uses_instance_rng(self):
        a = PhysicalTwinBridge(seed=123)
        b = PhysicalTwinBridge(seed=123)

        seq_a = [a.sync_step(0.4, 1) for _ in range(8)]
        seq_b = [b.sync_step(0.4, 1) for _ in range(8)]

        assert seq_a == seq_b
        assert all(np.isfinite(value) for value in seq_a)

    def test_tcp_mode_uses_json_line_contract(self, monkeypatch):
        writes: list[bytes] = []

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def sendall(self, data: bytes) -> None:
                writes.append(data)

            def makefile(self, mode: str, encoding: str):
                assert mode == "r"
                assert encoding == "utf-8"
                return iter(['{"v_mem": 0.875}\n'])

        def fake_create_connection(address, timeout):
            assert address == ("192.168.2.99", 5000)
            assert timeout == 0.25
            return FakeSocket()

        monkeypatch.setattr("socket.create_connection", fake_create_connection)
        bridge = PhysicalTwinBridge(mode="TCP", timeout_s=0.25)

        assert bridge.connected is False
        assert bridge.sync_step(0.5, 1) == pytest.approx(0.875)
        assert writes == [b'{"spike":1,"v_mem":0.5}\n']

    def test_tcp_mode_rejects_malformed_hardware_reply(self, monkeypatch):
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def sendall(self, data: bytes) -> None:
                pass

            def makefile(self, mode: str, encoding: str):
                return iter(["{}\n"])

        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: FakeSocket())
        bridge = PhysicalTwinBridge(mode="TCP")

        with pytest.raises(ValueError, match="missing numeric 'v_mem'"):
            bridge.sync_step(0.5, 0)


class TestDriverSourceHygiene:
    """Driver source suppressions must stay narrow and documented."""

    def test_pynq_optional_import_uses_narrow_type_ignore(self):
        source = pynq_driver.__loader__.get_source(pynq_driver.__name__)

        assert source is not None
        assert "type: ignore[import-not-found]" in source
        assert "type: ignore  # noqa" not in source


if __name__ == "__main__":
    test_driver_emulation_mode()
    test_driver_write_layer_params()
    test_driver_run_step()
    print("All driver tests passed!")

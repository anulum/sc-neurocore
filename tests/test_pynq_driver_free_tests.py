# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_pynq_driver.py

"""Module-level tests from former test_pynq_driver.py."""

from __future__ import annotations

from tests.pynq_driver_support import *  # noqa: F403

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

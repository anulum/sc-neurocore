# tests/test_pynq_driver.py
"""Tests for SC-NeuroCore PYNQ FPGA driver."""

import pytest
import numpy as np
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
    driver.write_layer_params(layer_id=1, params={'gain': 0.5, 'threshold': 1.0})
    driver.write_layer_params(layer_id=2, params={'gain': 2.0})

    print("Driver write_layer_params verified.")


def test_driver_run_step():
    """Verify run_step returns output in emulation mode."""
    driver = SC_NeuroCore_Driver(mode="EMULATION")

    input_vector = np.random.rand(16)
    output = driver.run_step(input_vector)

    assert output is not None
    assert len(output) == 16

    print("Driver run_step verified.")


def test_driver_hardware_mode_fails_without_fpga():
    """Verify hardware mode fails gracefully without FPGA."""
    with pytest.raises(RealityHardwareError):
        driver = SC_NeuroCore_Driver(mode="HARDWARE")


def test_driver_invalid_mode():
    """Verify invalid mode raises error."""
    with pytest.raises(ValueError):
        driver = SC_NeuroCore_Driver(mode="INVALID")


if __name__ == "__main__":
    test_driver_emulation_mode()
    test_driver_write_layer_params()
    test_driver_run_step()
    print("All driver tests passed!")

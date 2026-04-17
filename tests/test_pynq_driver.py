# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NeuroCore PYNQ FPGA driver

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
    driver.write_layer_params(layer_id=1, params={"gain": 0.5, "threshold": 1.0})
    driver.write_layer_params(layer_id=2, params={"gain": 2.0})

    print("Driver write_layer_params verified.")


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


if __name__ == "__main__":
    test_driver_emulation_mode()
    test_driver_write_layer_params()
    test_driver_run_step()
    print("All driver tests passed!")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerifyHardwareLink from former test_pynq_driver.py

"""Focused suite: TestVerifyHardwareLink from former test_pynq_driver.py."""

from __future__ import annotations

import pytest
import sys
import types
from typing import Never


class TestVerifyHardwareLink:
    """verify_link CLI smoke tests (closes task #31)."""

    def test_extras_false_fpga_only(self, capsys: pytest.CaptureFixture[str]) -> None:
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

    def test_extras_true_runs_all_three_probes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """extras=True (default) runs all three probes including the sibling-repo imports.

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

    def test_extras_default_is_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        """By default all three probes run."""
        from sc_neurocore.drivers.verify_hardware_link import verify_link

        verify_link()  # default
        out = capsys.readouterr().out
        assert "[3/3]" in out

    def test_no_sys_path_mutation(self) -> None:
        """verify_link must not mutate sys.path (closes the cross-repo bug)."""
        import sys

        from sc_neurocore.drivers.verify_hardware_link import verify_link

        before = list(sys.path)
        verify_link(extras=True)
        after = list(sys.path)
        assert before == after, "verify_link mutated sys.path"

    def test_fpga_probe_reports_success_when_driver_connects(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A driver that constructs cleanly drives the SUCCESS branch."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        monkeypatch.setattr(vhl, "SC_NeuroCore_Driver", lambda mode: object())
        vhl.verify_link(extras=False)
        out = capsys.readouterr().out
        assert "SUCCESS: PYNQ-Z2 Detected" in out

    def test_fpga_probe_reports_unexpected_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A raw OSError/RuntimeError (not RealityHardwareError) hits the ERROR branch."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        def boom(mode: str) -> Never:
            raise RuntimeError("bus fault")

        monkeypatch.setattr(vhl, "SC_NeuroCore_Driver", boom)
        vhl.verify_link(extras=False)
        out = capsys.readouterr().out
        assert "ERROR: Unexpected failure: bus fault" in out

    def test_genomic_probe_handles_present_but_unreachable_evo2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When Evo 2 is importable but its server is down, the probe warns cleanly."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        evo_mod = types.ModuleType("scpn_evo2_real_interface")

        class Evo2RealInterface:
            def connect(self) -> Never:
                raise ConnectionError("server down")

        vars(evo_mod)["Evo2RealInterface"] = Evo2RealInterface
        monkeypatch.setitem(sys.modules, "scpn_evo2_real_interface", evo_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Evo 2 Server unreachable" in out

    def test_robotics_probe_reports_opentrons_online(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A robot that answers is reported online."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self) -> bool:
                return True

        vars(ot_mod)["OpentronsVerifier"] = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Opentrons OT-2 Online" in out

    def test_robotics_probe_reports_opentrons_offline(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A robot that does not answer is reported offline."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self) -> bool:
                return False

        vars(ot_mod)["OpentronsVerifier"] = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "Robot offline" in out

    def test_robotics_probe_handles_opentrons_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A robot probe error is reported with its message."""
        import sc_neurocore.drivers.verify_hardware_link as vhl

        ot_mod = types.ModuleType("scpn_opentrions_verify")

        class OpentronsVerifier:
            def ping(self) -> Never:
                raise RuntimeError("robot fault")

        vars(ot_mod)["OpentronsVerifier"] = OpentronsVerifier
        monkeypatch.setitem(sys.modules, "scpn_opentrions_verify", ot_mod)

        vhl.verify_link(extras=True)
        out = capsys.readouterr().out
        assert "robot fault" in out

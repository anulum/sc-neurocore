# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for verify_hardware_link diagnostic tool

"""Tests for verify_hardware_link diagnostic tool."""

import importlib
from pathlib import Path

from sc_neurocore.drivers import verify_hardware_link
from sc_neurocore.drivers.verify_hardware_link import verify_link


def test_import():
    assert verify_link is not None


def test_verify_link_runs_without_crash(capsys):
    """verify_link should complete gracefully even without hardware."""
    verify_link()
    captured = capsys.readouterr()
    assert "DIAGNOSTIC COMPLETE" in captured.out


def test_verify_link_reports_no_pynq(capsys):
    """On x86 dev machines, FPGA check should fail gracefully."""
    verify_link()
    captured = capsys.readouterr()
    # Either "SUCCESS" (unlikely on CI) or graceful failure message
    assert "Checking FPGA" in captured.out


def test_verify_link_handles_missing_evo2(capsys):
    """Genomic interface import should fail gracefully."""
    verify_link()
    captured = capsys.readouterr()
    assert "Checking Genomic Interface" in captured.out


def test_verify_link_handles_missing_opentrons(capsys):
    """Robotics link check should fail gracefully."""
    verify_link()
    captured = capsys.readouterr()
    assert "Checking Robotics Link" in captured.out


def test_sys_path_append_uses_pathlib():
    """A4 audit: verify that path construction uses pathlib, not string concat."""
    source = importlib.util.find_spec("sc_neurocore.drivers.verify_hardware_link")
    assert source is not None
    src_path = Path(source.origin)
    assert src_path.exists()
    content = src_path.read_text(encoding="utf-8")
    assert "Path(__file__)" in content


def test_module_level_logger_exists():
    assert hasattr(verify_hardware_link, "logger")


def test_main_guard():
    """Module defines an if __name__ == '__main__' guard."""
    source = importlib.util.find_spec("sc_neurocore.drivers.verify_hardware_link")
    content = Path(source.origin).read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in content or "__name__ == '__main__'" in content

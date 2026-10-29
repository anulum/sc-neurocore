# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware catalogue registration contract tests

"""Exercise public hardware discovery against its actual registry owners."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sc_neurocore.hardware.device import DEVICE_CATALOG
from sc_neurocore.nir_bridge.hardware_targets import available_hardware_profiles
from sc_neurocore.nir_bridge.neuromorphic_adapters import SUPPORTED_ADAPTER_TARGETS
from sc_neurocore.studio.hardware_catalogue import hardware_catalogue
from sc_neurocore.studio.synthesis import synthesis_target_profiles


def test_catalogue_conserves_every_registered_target_and_exact_generation() -> None:
    """Retain NIR-only targets and distinguish incompatible hardware generations."""
    catalogue = hardware_catalogue()
    rows = {row.target_id: row for row in catalogue.entries}
    expected = {family.name.lower() for family in DEVICE_CATALOG}
    expected.update(profile.target_id for profile in available_hardware_profiles())
    expected.update("fpga:" + profile.target for profile in synthesis_target_profiles())
    expected.add("board:pynq_z2")
    assert set(rows) == expected
    assert list(rows) == sorted(expected)
    assert rows["brainscales2"].planning_profile
    assert rows["brainscales2"].nir_profile is None
    assert rows["brainscales3"].nir_profile is not None
    assert not rows["brainscales3"].planning_profile
    assert rows["dynap_se"].nir_profile is not None
    assert rows["board:pynq_z2"].driver is not None
    assert rows["fpga_generic"].kind == "fpga"
    assert rows["fpga_generic"].synthesis_profile is None
    for profile in available_hardware_profiles():
        entry = rows[profile.target_id]
        assert entry.nir_profile == profile
        assert entry.adapter_package == (profile.target_id in SUPPORTED_ADAPTER_TARGETS)
    assert all(
        row.evidence_level == "registration" and row.physical_evidence == ()
        for row in rows.values()
    )
    assert hardware_catalogue().model_dump_json() == catalogue.model_dump_json()


def test_catalogue_mutation_cannot_reconfigure_synthesis() -> None:
    """A reader's edited capacity snapshot cannot change future compiler inputs."""
    catalogue = hardware_catalogue()
    row = next(row for row in catalogue.entries if row.target_id == "fpga:ecp5")
    assert row.synthesis_profile is not None
    row.synthesis_profile.capacity.clear()
    new = next(row for row in hardware_catalogue().entries if row.target_id == "fpga:ecp5")
    assert new.synthesis_profile is not None
    assert new.synthesis_profile.capacity["luts"] == 24288


def test_catalogue_imports_without_vendor_sdks_or_optional_nir() -> None:
    """Actual missing optional dependencies do not prevent registration discovery."""
    source = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import importlib.abc
import sys
class NoVendorSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"nir", "pynq", "lava", "spynnaker", "akida", "torch", "onnx"}:
            raise ImportError("deliberately absent optional dependency: " + fullname)
sys.meta_path.insert(0, NoVendorSDK())
sys.path.insert(0, sys.argv[1])
from sc_neurocore.studio.hardware_catalogue import hardware_catalogue
def no_hardware_side_effect(event, args):
    if event in {"subprocess.Popen", "socket.connect", "os.mkdir", "os.remove", "os.rename"}:
        raise AssertionError("Registration discovery attempted " + event)
sys.addaudithook(no_hardware_side_effect)
catalogue = hardware_catalogue()
assert any(row.target_id == "dynap_se" for row in catalogue.entries)
assert all(row.physical_evidence == () for row in catalogue.entries)
assert "sc_neurocore.drivers.sc_neurocore_driver" not in sys.modules
assert not any(name.split(".")[0] in {"pynq", "lava", "spynnaker", "akida"} for name in sys.modules)
print(catalogue.model_dump_json())
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(source)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"schema_version":"studio.hardware-catalogue.v1"' in result.stdout

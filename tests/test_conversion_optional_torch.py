# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public conversion without the optional Torch dependency

"""Exercise the base package in an actual interpreter with no Torch import path."""

from pathlib import Path
import os
import subprocess
import sys

import coverage
import numpy as np


_PROBE = r"""
import importlib.util
import json
from pathlib import Path
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
assert importlib.util.find_spec("torch") is None
measurement = None
if sys.argv[3]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[3], branch=True, source=[sys.argv[1]], config_file=False)
    measurement.start()
import sc_neurocore.conversion as conversion
import sc_neurocore.conversion.ann_to_snn as implementation
assert Path(implementation.__file__).is_relative_to(Path(sys.argv[1]))
for call in (conversion.convert, conversion.replace_relu_with_qcfs):
    try:
        call(object())
    except ImportError as exc:
        assert "PyTorch required" in str(exc)
    else:
        raise AssertionError("conversion accepted absent Torch")
try:
    conversion.QCFSActivation
except ImportError as exc:
    assert "requires PyTorch" in str(exc)
else:
    raise AssertionError("QCFS accepted absent Torch")
network = conversion.ConvertedSNN([[[1.0]]], [None], [1.0], T=2)
assert network.run([1.0]).tolist() == [2.0]
assert network.replay([[[1.0]]], max_working_bytes=112).output.tolist() == [[1.0]]
if measurement is not None:
    measurement.stop()
    measurement.save()
print(json.dumps({"torch_absent": True, "public_refusals": 3, "origin": implementation.__file__}))
"""


def test_base_dependency_environment_refuses_torch_conversion(tmp_path: Path) -> None:
    """Load real NumPy and the public package with site imports disabled."""
    isolated = tmp_path / "packages"
    isolated.mkdir()
    isolated.joinpath("numpy").symlink_to(Path(np.__file__).parent, target_is_directory=True)
    isolated.joinpath("coverage").symlink_to(
        Path(coverage.__file__).parent, target_is_directory=True
    )
    source = Path(__file__).resolve().parents[1] / "src"
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = parent_data + "-no-torch" if parent_data else ""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _PROBE, str(source), str(isolated), child_data],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert '"torch_absent": true' in result.stdout
    assert '"public_refusals": 3' in result.stdout

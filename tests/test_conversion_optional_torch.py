# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public conversion without the optional Torch dependency

"""Exercise the base package in an actual interpreter with no Torch import path."""

from pathlib import Path
import importlib.metadata
import importlib.util
import os
import subprocess
import sys

import coverage
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parents[1]


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


def _base_distributions() -> list[str]:
    """Return the declared base requirements that apply to this interpreter."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = []
    for line in project["dependencies"]:
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            names.append(requirement.name)
    return names


def _link_distribution(distribution: str, isolated: Path) -> None:
    """Expose every top-level import of one installed distribution, and nothing else.

    The names come from the distribution's installed file list: Python 3.10's
    ``packages_distributions`` reads only ``top_level.txt``, which NumPy lacks.
    """
    modules = set()
    for file in importlib.metadata.distribution(distribution).files or ():
        head = file.parts[0]
        if len(file.parts) == 1:
            head = head.split(".", 1)[0] if head.endswith((".py", ".so", ".pyd")) else ""
        if head.isidentifier() and head != "__pycache__":
            modules.add(head)
    assert modules, f"declared base dependency {distribution} installs no importable module"
    for module in sorted(modules):
        spec = importlib.util.find_spec(module)
        assert spec is not None, f"{distribution} declares unimportable {module}"
        if spec.submodule_search_locations:
            location = Path(next(iter(spec.submodule_search_locations)))
            isolated.joinpath(module).symlink_to(location, target_is_directory=True)
        elif spec.origin is not None:
            origin = Path(spec.origin)
            isolated.joinpath(origin.name).symlink_to(origin)


def test_base_dependency_environment_refuses_torch_conversion(tmp_path: Path) -> None:
    """Load the declared base dependencies and the package with site imports disabled.

    The isolated path holds exactly the ``[project].dependencies`` that apply to
    the running interpreter (for example ``tomli`` below Python 3.11), plus
    coverage for measurement, so a base install that cannot import the public
    conversion API fails here rather than for a user.
    """
    isolated = tmp_path / "packages"
    isolated.mkdir()
    base = _base_distributions()
    assert "torch" not in base
    for distribution in base:
        _link_distribution(distribution, isolated)
    isolated.joinpath("coverage").symlink_to(
        Path(coverage.__file__).parent, target_is_directory=True
    )
    source = _ROOT / "src"
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = parent_data + "-no-torch" if parent_data else ""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _PROBE, str(source), str(isolated), child_data],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"torch_absent": true' in result.stdout
    assert '"public_refusals": 3' in result.stdout

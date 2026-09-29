# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed-wheel QCFS runtime acceptance

"""Build every QCFS runtime from an installed wheel and require the five-way bit parity."""

import os
import subprocess
from pathlib import Path

from tests.test_conversion_if_distribution import installed
from tests.test_conversion_qcfs_native import _PARITY, qcfs_environment, qcfs_libraries
from tests.test_studio_distribution import distribution_source, installation_environment

__all__ = ["distribution_source", "installed", "qcfs_environment"]


def test_installed_wheel_builds_and_matches_every_qcfs_runtime(
    installed: Path, qcfs_environment: dict[str, str], tmp_path: Path
) -> None:
    """The wheel's own Rust, Go, Mojo and Julia QCFS sources reproduce the NumPy bits."""
    python = installation_environment(installed, tmp_path / "environment")
    libraries = qcfs_libraries(installed / "sc_neurocore/accel", tmp_path / "native")
    settings = {key: value for key, value in qcfs_environment.items() if key != "PYTHONPATH"}
    settings.update({name: str(path) for name, path in libraries.items()})
    parent_data = os.environ.get("COVERAGE_FILE", "")
    probe = (
        "import sys\nfrom pathlib import Path\nimport sc_neurocore\n"
        "assert Path(sc_neurocore.__file__).is_relative_to(Path(sys.argv[3])), sc_neurocore.__file__\n"
    ) + _PARITY
    result = subprocess.run(
        [
            str(python),
            "-c",
            probe,
            f"{parent_data}-installed-qcfs" if parent_data else "",
            str(installed / "sc_neurocore/conversion"),
            str(installed),
        ],
        cwd=tmp_path,
        env=settings,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "five runtimes bit-identical over" in result.stdout

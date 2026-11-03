# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit native-gate Python counter custody

"""Keep own Python tracing explicit, source-bound and separate from native Go counters.

Only the standard coverage ``_exit`` patch is enabled. No subprocess patch,
global bootstrap or inherited coverage configuration is installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import cast

_HEADER = "\n".join(Path(__file__).read_text(encoding="utf-8").splitlines()[:7]) + "\n\n"


def expected_sources(repository: Path) -> dict[str, str]:
    """Pin every actual introduced gate tool and its complete ABI consumer.

    Parameters
    ----------
    repository : pathlib.Path
        Actual current source checkout supplying the gate.

    Returns
    -------
    dict
        Repository-relative source names and complete SHA256 digests.
    """
    paths = [
        *repository.joinpath("tools").glob("go*coverage*.py"),
        repository / "tests/native_go_coverage_consumer.py",
    ]
    if len(paths) < 14 or any(path.is_symlink() for path in paths):
        raise ValueError("The complete actual gate Python source cohort is unavailable")
    return {
        path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def config_text(repository: Path, directory: Path, other_roots: tuple[str, ...] = ()) -> str:
    """Describe the exact branch/counter scope without exclusions or inherited settings."""
    if any(char in str(path) for path in (repository, directory) for char in "\n\r$*?[]") or any(
        char in root for root in other_roots for char in "\n\r$*?[]"
    ):
        raise ValueError("Coverage paths contain ambiguous configuration or pattern characters")
    names = expected_sources(repository)
    include = "\n".join("    " + (repository / name).as_posix() for name in names)
    text = (
        _HEADER
        + "[run]\nbranch = true\nparallel = true\npatch = _exit\n"
        + "data_file = "
        + (directory / ".coverage").as_posix()
        + "\ninclude =\n"
        + include
        + "\n[report]\nfail_under = 100\nshow_missing = true\n"
        + "exclude_lines =\npartial_branches =\n"
    )
    if other_roots:
        text += "[paths]\nproject =\n    " + repository.as_posix() + "\n"
        text += "".join("    " + root.replace("\\", "/") + "\n" for root in other_roots)
    return text


def write_config(repository: Path, directory: Path) -> Path:
    """Create one exclusive trace allocation and its actual source/platform manifest."""
    repository = repository.resolve(strict=True)
    if not directory.is_absolute() or directory != directory.resolve():
        raise ValueError("Declare the actual absolute trace allocation without aliases")
    text = config_text(repository, directory)
    pin = re.search(
        r"^coverage==([^\s]+)",
        (repository / "requirements/ci-dev.txt").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    version = importlib.metadata.version("coverage")
    if pin is None or version != pin[1]:
        raise RuntimeError("Actual coverage version differs from the existing locked CI pin")
    directory.mkdir()
    path = directory / "go-mat.coveragerc"
    path.write_text(text, encoding="utf-8")
    (directory / "source-platform.json").write_text(
        json.dumps(
            {
                "repository": str(repository),
                "platform": sys.platform,
                "coverage_version": version,
                "sources": expected_sources(repository),
                "coverage_pin_sha256": hashlib.sha256(
                    (repository / "requirements/ci-dev.txt").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def validate_config(path: Path, repository: Path) -> str:
    """Refuse any changed tracing scope, source, platform or nonstandard patch setting."""
    if (
        not path.is_absolute()
        or path != path.resolve(strict=True)
        or path.is_symlink()
        or path.name != "go-mat.coveragerc"
    ):
        raise ValueError("Use the explicit actual gate-owned coverage configuration")
    text = path.read_text(encoding="utf-8")
    metadata = json.loads((path.parent / "source-platform.json").read_text(encoding="utf-8"))
    if text != config_text(repository, path.parent):
        raise ValueError("The exact own Python statement/branch configuration changed")
    if (
        metadata.get("repository") != str(repository)
        or metadata.get("platform") != sys.platform
        or metadata.get("sources") != expected_sources(repository)
        or metadata.get("coverage_version") != importlib.metadata.version("coverage")
        or metadata.get("coverage_pin_sha256")
        != hashlib.sha256((repository / "requirements/ci-dev.txt").read_bytes()).hexdigest()
    ):
        raise ValueError("The trace source/platform/version manifest changed")
    return hashlib.sha256(text.encode()).hexdigest()


def write_bootstrap(job: Path, repository: Path, config: Path) -> Path:
    """Generate only this allocation's real-child tracing and native-observer entry points."""
    validate_config(config, repository)
    startup = (
        _HEADER
        + "import atexit\nimport coverage\n"
        + "_gate_coverage = coverage.Coverage(config_file="
        + repr(str(config))
        + ")\n"
        + "_gate_coverage.start()\n"
        + "def _save_gate_coverage():\n    _gate_coverage.stop()\n    _gate_coverage.save()\n"
        + "atexit.register(_save_gate_coverage)\nimport sys\n"
        + "sys.path.insert(0,"
        + repr(str(repository))
        + ")\n"
        + "from tools.go_coverage_images import bootstrap\nbootstrap()\n"
    )
    preload = (
        _HEADER
        + "import ctypes\nimport sys\nfrom pathlib import Path\n"
        + "sys.path.insert(0,sys.argv[1])\n"
        + "from tools.go_coverage_images import bootstrap, read_images\nbootstrap()\n"
        + "job=Path(sys.argv[2])\n"
        + "for image in read_images(job/sys.argv[3],job).values():\n"
        + "    if image.backend=='go':\n        ctypes.CDLL(str(image.path))\n"
    )
    for name, content in (("sitecustomize.py", startup), ("preload.py", preload)):
        with (job / "bootstrap" / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return job / "bootstrap/preload.py"


def union_config(
    repository: Path,
    directory: Path,
    manifests: tuple[Path, ...],
    required_platforms: frozenset[str],
) -> Path:
    """Bind counter aliases after exact source, version and platform conservation."""
    expected = expected_sources(repository)
    roots: list[str] = []
    platforms: set[str] = set()
    version = importlib.metadata.version("coverage")
    pin_digest = hashlib.sha256((repository / "requirements/ci-dev.txt").read_bytes()).hexdigest()
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        root, platform = data.get("repository"), data.get("platform")
        if (
            not isinstance(root, str)
            or not root
            or not isinstance(platform, str)
            or platform in platforms
            or data.get("sources") != expected
            or data.get("coverage_version") != version
            or data.get("coverage_pin_sha256") != pin_digest
        ):
            raise ValueError(
                "Native gate counter cohorts have different source/version or duplicate platform"
            )
        roots.append(root)
        platforms.add(platform)
    if not platforms or platforms != required_platforms:
        raise ValueError("Actual native gate counter platforms differ from the required cohort")
    text = config_text(repository, directory, tuple(roots))
    directory.mkdir()
    output = directory / "go-mat-union.coveragerc"
    output.write_text(text, encoding="utf-8")
    return output


def check_report_membership(report: Path, repository: Path) -> dict[str, object]:
    """Require the entire actual source cohort; the standard coverage CLI gates counters."""
    raw: object = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise ValueError("Actual coverage JSON has no source-file mapping")
    files = cast(dict[str, object], raw["files"])
    if any(
        not isinstance(info, dict) or info.get("excluded_lines") != [] for info in files.values()
    ):
        raise ValueError(
            "Actual own Python coverage contains missing entries or excluded statements"
        )
    expected = set(expected_sources(repository))
    actual = set()
    for name in files:
        path = Path(name)
        resolved = (
            path.resolve(strict=True)
            if path.is_absolute()
            else (repository / path).resolve(strict=True)
        )
        if not resolved.is_relative_to(repository):
            raise ValueError("Actual coverage JSON contains a foreign source")
        actual.add(resolved.relative_to(repository).as_posix())
    if actual != expected or len(actual) != len(files):
        raise ValueError("Actual own Python coverage source cohort is missing, extra or duplicated")
    return {
        "actual_sources": sorted(actual),
        "required_sources": sorted(expected),
        "counter_qualification_claimed": False,
        "counter_gate": "standard coverage report --fail-under=100, no exclusions",
    }

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense IF benchmark source and artifact identity

"""Bind measured public/native implementations and explicitly configured artifacts."""

import hashlib
import os
from pathlib import Path


def source_digests(resources: Path | None = None) -> dict[str, str]:
    """Bind the benchmark and all maintained runtime counterparts to current source bytes.

    The public and native owners are always read from the imported ``sc_neurocore``
    package, so a report binds the code that actually ran, whether that package is a
    source checkout or an installed wheel.

    Parameters
    ----------
    resources : Path or None
        Directory holding the executing comparison scripts. None resolves the
        installed package's ``conversion/benchmark_resources``, or the owning
        checkout's ``benchmarks`` when the package is a source tree.

    Returns
    -------
    dict of str to str
        Repository-relative owning source paths and their SHA-256 digests.

    Raises
    ------
    OSError
        The build declaration beside the comparison scripts cannot be read.
    """
    package = Path(__file__).resolve().parents[1]
    if resources is None:
        resources = package / "conversion/benchmark_resources"
        if not resources.is_dir():
            resources = package.parents[1] / "benchmarks"
    declaration = resources / "pyproject.toml"
    if not declaration.is_file():
        declaration = resources.parent / "pyproject.toml"
    sources = sorted((package / "conversion").glob("*.py"))
    for language, pattern in (
        ("mojo/kernels", "ann_to_snn*.mojo"),
        ("julia/conversion", "*.jl"),
        ("go/conversion", "*.go"),
        ("go/conversion/cshared", "*.go"),
        ("rust/safety", "ann_to_snn*.rs"),
        ("rust/safety/if_native/src", "*.rs"),
    ):
        sources += sorted((package / "accel" / language).glob(pattern))
    sources += [
        package / name
        for name in (
            "accel/go/go.mod",
            "accel/go/conversion/cshared/abi.h",
            "accel/rust/safety/if_native/Cargo.toml",
            "accel/rust/safety/if_native/Cargo.lock",
        )
    ]
    result = {
        "src/sc_neurocore/" + str(p.relative_to(package)): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sources
    }
    result.update(
        {
            "benchmarks/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(resources.glob("*ann_to_snn_replay*.py"))
        }
    )
    result["pyproject.toml"] = hashlib.sha256(declaration.read_bytes()).hexdigest()
    return result


def configured_artifacts(
    backends: tuple[str, ...] = ("rust", "go", "mojo", "julia"),
) -> dict[str, str]:
    """Require explicit installed native inputs and bind their actual bytes without building.

    Parameters
    ----------
    backends : tuple of str
        Configured providers whose actual artifact bytes must be bound.

    Returns
    -------
    dict of str to str
        Declared native artifact names, Julia executable and locked-project digests.

    Raises
    ------
    KeyError, OSError
        Required owner configuration or its actual installed file is absent.
    """
    artifacts = {}
    for backend in backends:
        if backend == "julia":
            continue
        backend = backend.upper()
        path = Path(os.environ[f"SC_NEUROCORE_IF_{backend}_LIB"]).expanduser().resolve(strict=True)
        artifacts[f"SC_NEUROCORE_IF_{backend}_LIB"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "julia" not in backends:
        return artifacts
    executable = Path(os.environ["PYTHON_JULIACALL_EXE"]).resolve(strict=True)
    project = Path(os.environ["PYTHON_JULIACALL_PROJECT"]).resolve(strict=True)
    for label, path in (
        ("PYTHON_JULIACALL_EXE", executable),
        ("PYTHON_JULIACALL_PROJECT/Project.toml", project / "Project.toml"),
        ("PYTHON_JULIACALL_PROJECT/Manifest.toml", project / "Manifest.toml"),
    ):
        artifacts[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts

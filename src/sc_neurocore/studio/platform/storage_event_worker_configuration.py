# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Operator event inputs for isolated workers

"""Declare trusted recording and native runtime paths outside job requests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


def _normalised(path: Path) -> None:
    """Refuse relative or lexically noncanonical operator paths."""
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError("event worker paths must be absolute and normalised")


class EventJuliaRuntime(BaseModel):
    """An explicit installed JuliaCall runtime, with one thread and signal policy.

    The operator must provision a compatible PythonCall project beforehand.
    Declaring this object opts into Julia decoding; no package is installed.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    executable: Path
    project: Path
    handle_signals: Literal["yes", "no"]

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        """Require existing operator executable and project paths."""
        _normalised(self.executable)
        _normalised(self.project)
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError("event worker Julia executable is unavailable")
        if not self.project.is_dir() or not (self.project / "Project.toml").is_file():
            raise ValueError("event worker Julia project is unavailable")
        return self


class EventWorkerConfiguration(BaseModel):
    """Operator-owned event root, input budget and optional native decoders.

    Paths must be readable by the configured compute identity. Validation
    checks their current availability to the launcher; it does not prove
    worker access, dependency compatibility or immutable file custody.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    dataset_root: Path
    input_max_bytes: Annotated[int, Field(gt=0)] = 64 * 1024 * 1024
    rust_library: Path | None = None
    mojo_library: Path | None = None
    go_library: Path | None = None
    dvs_go_executable: Path | None = None
    dvs_rust_executable: Path | None = None
    dvs_julia_executable: Path | None = None
    dvs_mojo_executable: Path | None = None
    shd_go_library: Path | None = None
    shd_rust_library: Path | None = None
    shd_mojo_executable: Path | None = None
    shd_mojo_hdf5_library: Path | None = None
    shd_julia_executable: Path | None = None
    shd_julia_hdf5_library: Path | None = None
    julia: EventJuliaRuntime | None = None

    @model_validator(mode="after")
    def validate_recordings(self) -> Self:
        """Require an existing root and existing absolute native library files."""
        _normalised(self.dataset_root)
        if not self.dataset_root.is_dir():
            raise ValueError("event worker dataset root is unavailable")
        for library in (
            self.rust_library,
            self.mojo_library,
            self.go_library,
            self.shd_go_library,
            self.shd_rust_library,
            self.shd_julia_hdf5_library,
            self.shd_mojo_hdf5_library,
        ):
            if library is not None:
                _normalised(library)
                if not library.is_file():
                    raise ValueError("event worker native library is unavailable")
        for backend, executable, library in (
            ("Julia", self.shd_julia_executable, self.shd_julia_hdf5_library),
            ("Mojo", self.shd_mojo_executable, self.shd_mojo_hdf5_library),
        ):
            if executable is not None:
                _normalised(executable)
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise ValueError(f"SHD worker {backend} executable is unavailable")
            if library is not None and executable is None:
                raise ValueError(f"SHD {backend} HDF5 selection requires a {backend} executable")
        if (
            sum(
                executable is not None
                for executable in (
                    self.dvs_go_executable,
                    self.dvs_rust_executable,
                    self.dvs_julia_executable,
                    self.dvs_mojo_executable,
                )
            )
            > 1
        ):
            raise ValueError("DVS worker requires one native executable selection")
        for backend, executable in (
            ("Go", self.dvs_go_executable),
            ("Rust", self.dvs_rust_executable),
            ("Julia", self.dvs_julia_executable),
            ("Mojo", self.dvs_mojo_executable),
        ):
            if executable is not None:
                _normalised(executable)
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise ValueError(f"DVS worker {backend} executable is unavailable")
        return self

    def environment(self) -> dict[str, str]:
        """Build only the declared event settings for the fixed worker environment.

        Returns
        -------
        dict
            Recording root, input limit and selected native runtime settings.
            Neither API request fields nor the launcher's inherited environment
            participate in this result.
        """
        environment = {
            "SC_NEUROCORE_STUDIO_DATASET_ROOT": str(self.dataset_root),
            "SC_NEUROCORE_STUDIO_EVENT_INPUT_MAX_BYTES": str(self.input_max_bytes),
        }
        for backend, library in (
            ("RUST", self.rust_library),
            ("MOJO", self.mojo_library),
            ("GO", self.go_library),
        ):
            if library is not None:
                environment[f"SC_NEUROCORE_DATASET_{backend}_LIBRARY"] = str(library)
        for backend, library in (("GO", self.shd_go_library), ("RUST", self.shd_rust_library)):
            if library is not None:
                environment[f"SC_NEUROCORE_SHD_{backend}_LIBRARY"] = str(library)
        for backend, executable, library in (
            ("JULIA", self.shd_julia_executable, self.shd_julia_hdf5_library),
            ("MOJO", self.shd_mojo_executable, self.shd_mojo_hdf5_library),
        ):
            if executable is not None:
                environment[f"SC_NEUROCORE_SHD_{backend}_EXE"] = str(executable)
            if library is not None:
                environment[f"SC_NEUROCORE_SHD_{backend}_HDF5_LIBRARY"] = str(library)
        for backend, executable in (
            ("GO", self.dvs_go_executable),
            ("RUST", self.dvs_rust_executable),
            ("JULIA", self.dvs_julia_executable),
            ("MOJO", self.dvs_mojo_executable),
        ):
            if executable is not None:
                environment[f"SC_NEUROCORE_DVS_{backend}_EXE"] = str(executable)
        if self.julia is not None:
            environment.update(
                SC_NEUROCORE_DATASET_JULIA_ENABLED="1",
                PYTHON_JULIACALL_EXE=str(self.julia.executable),
                PYTHON_JULIACALL_PROJECT=str(self.julia.project),
                PYTHON_JULIACALL_THREADS="1",
                PYTHON_JULIACALL_HANDLE_SIGNALS=self.julia.handle_signals,
            )
        return environment

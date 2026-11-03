# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gate-owned Go MAT image and counter observation

"""Record actual MAT image loads and require native snapshots before process exit."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import TextIO

from tools.go_coverage_image_inputs import (
    Image as Image,
    executable_image_maps,
    original_abi_launch_arguments,
    read_images as read_images,
    validate_audited_image,
    validate_existing_observer,
    validate_invalid_snapshot_status,
    validate_snapshot_output,
)

from tools.go_coverage_environment import (
    child_declaration,
    environment_digest,
    environment_mapping,
    read_child_environment,
    terminal_environment,
)

_ALLOWED_ENVIRONMENT = frozenset(
    {
        "COVERAGE_RUN",
        "COLUMNS",
        "LINES",
        "GOMAXPROCS",
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "GOCOVERDIR",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "NEUROCORE_GO_COVERAGE_JOB",
        "NEUROCORE_GO_COVERAGE_PHASE",
        "NEUROCORE_GO_COVERAGE_MANIFEST",
    }
)
_PHASES = frozenset(
    {
        "before",
        "tests",
        "contracts",
        "wrong-digest",
        "missing-image",
        "normal-image",
        "uncovered-tag",
    }
)

_OWNER: CounterObserver | None = None


class CounterObserver:
    """Hold per-process load observations and collect actual Go counter snapshots."""

    __cantrace__: bool = True

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        """Deliver real audit events through CPython's traceable callable protocol."""
        self.audit(event, arguments)

    def __init__(self, job: Path, images: dict[Path, Image], phase: str) -> None:
        """Create a fresh PID record before registering any runtime hooks."""
        self.images = images
        self.job = job
        self.phase = phase
        self.loaded: dict[Path, tuple[int, int]] = {}
        self.flushing = False
        output = job / "process-counters" / phase
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / str(os.getpid())
        self.output.mkdir()
        self.stream: TextIO = (self.output / "observations.jsonl").open("x")
        self.record(
            {
                "event": "registered",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "argv": sys.argv,
                "interpreter": running_interpreter_identity(),
            }
        )

    def record(self, row: dict[str, object]) -> None:
        """Persist each observation immediately so partial process failures remain visible."""
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()

    def audit(self, event: str, arguments: tuple[object, ...]) -> None:
        """Validate actual family loads while leaving unrelated runtime loads alone."""
        if event == "subprocess.Popen":
            self.record_child_launch(arguments)
            return
        if self.flushing or event != "ctypes.dlopen" or not arguments:
            return
        value = arguments[0]
        if not isinstance(value, (str, bytes)):
            return
        path = Path(os.fsdecode(value))
        if path.name not in {"libmat.so", "libsc_resetting_mat.so"}:
            return
        actual = path.resolve(strict=True)
        image = self.images.get(actual)
        if image is None:
            raise RuntimeError("Unadmitted MAT family image")
        with actual.open("rb") as source:
            before = os.fstat(source.fileno())
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = os.fstat(source.fileno())
        identity = validate_audited_image(image, before, after, digest, self.loaded.get(actual))
        self.loaded[actual] = identity
        self.record(
            {
                "event": "pre_dlopen",
                "pid": os.getpid(),
                "path": str(actual),
                "sha256": digest,
                "device": before.st_dev,
                "inode": before.st_ino,
                "size": before.st_size,
                "backend": image.backend,
                "model": image.model,
            }
        )

    def record_child_launch(self, arguments: tuple[object, ...]) -> None:
        """Retain actual original crash-fenced child launches for independent reconciliation."""
        strings = original_abi_launch_arguments(arguments, self.phase)
        if strings is None:
            return
        base = environment_mapping(json.loads((self.job / "environments/tests.json").read_text()))
        declaration = child_declaration(self.job, base, dict(os.environ))
        self.record(
            {
                "event": "abi_child_launch",
                "parent_pid": os.getpid(),
                "executable": strings[0],
                "expected_child_argv": ["-c", *strings[4:]],
                "audit_arguments": list(arguments),
                **declaration,
            }
        )

    def mapped_identity(self, image: Image) -> list[str]:
        """Require an executable mapping of the observed backing inode at snapshot time."""
        return executable_image_maps(self.loaded[image.path], Path("/proc/self/maps").read_text())

    def finish(self) -> None:
        """Fail the process if any loaded Go image cannot supply a genuine snapshot."""
        self.flushing = True
        try:
            for path in self.loaded:
                image = self.images[path]
                image.verify_backing(self.loaded[path])
                mapped = self.mapped_identity(image)
                if image.backend != "go":
                    self.record(
                        {"event": "retained_mojo_mapping", "path": str(path), "maps": mapped}
                    )
                    continue
                library = ctypes.CDLL(str(path))
                writer = library.write_coverage_snapshot
                writer.argtypes = [ctypes.c_char_p]
                writer.restype = ctypes.c_int
                refused = int(writer(None))
                self.record(
                    {
                        "event": "invalid_snapshot_path",
                        "pid": os.getpid(),
                        "path": str(path),
                        "status": refused,
                    }
                )
                validate_invalid_snapshot_status(refused)
                # The second cumulative snapshot includes the first writer's
                # completed success return. Counters are never unique-call totals.
                for capture in ("first", "final"):
                    destination = self.output / image.model / capture
                    destination.mkdir(parents=True)
                    status = int(writer(os.fsencode(destination)))
                    self.record(
                        {
                            "event": "snapshot",
                            "capture": capture,
                            "pid": os.getpid(),
                            "path": str(path),
                            "sha256": image.sha256,
                            "status": status,
                            "maps": mapped,
                            "destination": str(destination),
                            "files": sorted(child.name for child in destination.iterdir()),
                        }
                    )
                    validate_snapshot_output(destination, status)
            self.record({"event": "finished", "pid": os.getpid()})
            self.stream.close()
        except BaseException as error:
            try:
                self.record({"event": "snapshot_failure", "pid": os.getpid(), "error": repr(error)})
                self.stream.close()
            finally:
                os._exit(97)


def running_interpreter_identity() -> dict[str, str]:
    """Describe the actual running Python executable, prefix and binary bytes."""
    executable = Path(sys.executable).resolve(strict=True)
    with executable.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "executable": str(executable),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "sha256": digest,
        "version": sys.version,
    }


def register(job: Path, phase: str, manifest_name: str = "images.json") -> None:
    """Attach only inside the gate's explicitly closed subprocess environment.

    Parameters
    ----------
    job : pathlib.Path
        Fresh gate allocation containing reviewed environment and image inputs.
    phase : str
        Named observation phase written by the gate before launching the child.
    manifest_name : str, optional
        Gate-owned image manifest; fault variants remain separate immutable files.

    Raises
    ------
    ValueError
        A phase, manifest or environment key/value is outside the explicit contract.
    RuntimeError
        The interpreter, actual environment or declared allocation differs.
    """
    global _OWNER
    if sys.flags.optimize or not job.is_absolute() or job.is_symlink():
        raise RuntimeError("An explicit allocation and unoptimized Python are required")
    job = job.resolve(strict=True)
    if phase not in _PHASES or manifest_name not in {
        "images.json",
        "images-wrong-digest.json",
        "images-missing-image.json",
        "images-normal.json",
        "images-uncovered-tag.json",
    }:
        raise ValueError("Unknown native coverage observation phase or manifest")
    expected = environment_mapping(
        json.loads((job / "environments" / (phase + ".json")).read_text())
    )
    if not set(expected).issubset(_ALLOWED_ENVIRONMENT):
        raise ValueError("An unreviewed consumer environment key is present")
    if phase == "tests" and sys.argv[:1] == ["-c"]:
        expected = read_child_environment(job, expected, os.getppid(), sys.argv)
    native = {
        item.split(b"=", 1)[0].decode(): item.split(b"=", 1)[1].decode()
        for item in Path("/proc/self/environ").read_bytes().split(b"\0")
        if item
    }
    if native != expected or dict(os.environ) != expected:
        raise RuntimeError("Actual consumer keys or values differ from the closed environment")
    terminal = terminal_environment()
    if terminal != {key: expected.get(key) for key in ("COLUMNS", "LINES")}:
        raise RuntimeError("Actual Readline terminal metadata changed after declaration")
    if (
        expected.get("NEUROCORE_GO_COVERAGE_JOB") != str(job)
        or expected.get("NEUROCORE_GO_COVERAGE_PHASE") != phase
        or expected.get("NEUROCORE_GO_COVERAGE_MANIFEST") != manifest_name
    ):
        raise RuntimeError("Consumer task, phase or manifest identity differs")
    if running_interpreter_identity() != json.loads((job / "python.json").read_text()):
        raise RuntimeError("Actual Python executable or prefix differs from the gate producer")
    if _OWNER is not None:
        validate_existing_observer(_OWNER.job, _OWNER.phase, job, phase)
        return
    owner = CounterObserver(job, read_images(job / manifest_name, job), phase)
    owner.record(
        {
            "event": "closed_environment",
            "keys": sorted(native),
            "environment_sha256": environment_digest(expected),
            "terminal_environment": terminal,
        }
    )
    _OWNER = owner
    sys.addaudithook(owner)
    atexit.register(owner.finish)


def bootstrap() -> None:
    """Enter from the gate-owned temporary child bootstrap, never a global install."""
    try:
        register(
            Path(os.environ["NEUROCORE_GO_COVERAGE_JOB"]),
            os.environ["NEUROCORE_GO_COVERAGE_PHASE"],
            os.environ["NEUROCORE_GO_COVERAGE_MANIFEST"],
        )
    except BaseException as error:
        print("Native coverage bootstrap refused: " + repr(error), file=sys.stderr, flush=True)
        os._exit(96)

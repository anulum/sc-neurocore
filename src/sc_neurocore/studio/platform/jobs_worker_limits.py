# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Embedded Studio worker resource limits

"""Bound a POSIX Studio worker before it imports a job's task module."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_WORKER_MAX_OPEN_FILES = 4096
DEFAULT_WORKER_MAX_FILE_BYTES = 8 * 1024**3


@dataclass(frozen=True, slots=True)
class StudioWorkerLimits:
    """Per-process ceilings for embedded Studio jobs.

    Parameters
    ----------
    max_data_bytes:
        Private writable allocation ceiling in bytes.
    max_open_files:
        Maximum number of open file descriptors.
    max_file_bytes:
        Maximum size in bytes of any file written by the worker.
    max_cpu_seconds:
        CPU seconds per process. ``None`` derives a ceiling from the job's
        wall-clock timeout and the host CPU count.
    """

    max_data_bytes: int
    max_open_files: int
    max_file_bytes: int
    max_cpu_seconds: int | None = None

    def __post_init__(self) -> None:
        """Reject nonpositive, fractional and boolean resource ceilings."""
        for name in ("max_data_bytes", "max_open_files", "max_file_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Studio worker limit {name} must be a positive integer.")
        if self.max_cpu_seconds is not None and (
            isinstance(self.max_cpu_seconds, bool)
            or not isinstance(self.max_cpu_seconds, int)
            or self.max_cpu_seconds <= 0
        ):
            raise ValueError("Studio worker limit max_cpu_seconds must be a positive integer.")

    @classmethod
    def for_host(cls) -> StudioWorkerLimits:
        """Use half of physical RAM and bounded descriptor and file ceilings.

        Returns
        -------
        StudioWorkerLimits
            Defaults for a POSIX host.
        """
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return cls(
            max_data_bytes=max(1, memory // 2),
            max_open_files=DEFAULT_WORKER_MAX_OPEN_FILES,
            max_file_bytes=DEFAULT_WORKER_MAX_FILE_BYTES,
        )

    def worker_arguments(self, timeout_seconds: float) -> list[str]:
        """Encode limits for the worker launched with this job timeout.

        Parameters
        ----------
        timeout_seconds:
            Finite positive wall-clock timeout of the job.

        Returns
        -------
        list[str]
            Complete option and value pairs for the process worker.

        Raises
        ------
        ValueError
            If the timeout is nonpositive or not finite.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Studio worker timeout must be finite and positive.")
        cpu_seconds = self.max_cpu_seconds or max(
            1, math.ceil(timeout_seconds * (os.cpu_count() or 1))
        )
        return [
            "--max-data-bytes",
            str(self.max_data_bytes),
            "--max-cpu-seconds",
            str(cpu_seconds),
            "--max-open-files",
            str(self.max_open_files),
            "--max-file-bytes",
            str(self.max_file_bytes),
        ]


def apply_worker_limits(
    *, max_data_bytes: int, max_cpu_seconds: int, max_open_files: int, max_file_bytes: int
) -> None:
    """Set unraisable POSIX limits in the worker before loading task code.

    An inherited hard ceiling is never raised. CPU has a one-second soft-to-hard
    interval so ``SIGXCPU`` can report exhaustion before the kernel kills the
    worker. No per-UID ``RLIMIT_NPROC`` is set.

    Parameters
    ----------
    max_data_bytes:
        Private writable allocation ceiling in bytes.
    max_cpu_seconds:
        CPU seconds before ``SIGXCPU``.
    max_open_files:
        Maximum open file descriptors.
    max_file_bytes:
        Maximum bytes in one output file.
    """
    import resource

    for limit, soft, hard in (
        (resource.RLIMIT_DATA, max_data_bytes, max_data_bytes),
        (resource.RLIMIT_CPU, max_cpu_seconds, max_cpu_seconds + 1),
        (resource.RLIMIT_NOFILE, max_open_files, max_open_files),
        (resource.RLIMIT_FSIZE, max_file_bytes, max_file_bytes),
        (resource.RLIMIT_CORE, 0, 0),
    ):
        _, inherited_hard = resource.getrlimit(limit)
        if inherited_hard != resource.RLIM_INFINITY:
            soft = min(soft, inherited_hard)
            hard = min(hard, inherited_hard)
        resource.setrlimit(limit, (soft, hard))

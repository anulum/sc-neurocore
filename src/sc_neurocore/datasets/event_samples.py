# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Individual manifest-bound event samples

"""Read one event recording at a time, without loading a complete corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.accel.shd_recordings import read_shd_recording
from sc_neurocore.datasets.loaders import _parse_dvs_npy, _parse_nmnist_bin
from sc_neurocore.datasets.manifest import SampleRecord


def read_event_sample(root: Path, dataset: str, sample: SampleRecord) -> np.ndarray[Any, Any]:
    """Read a manifest sample as spatial events with millisecond timestamps.

    Parameters
    ----------
    root:
        Operator dataset root whose manifest has already been verified.
    dataset:
        ``nmnist``, ``shd`` or ``dvs_cifar10``.
    sample:
        Sample location in that verified manifest. SHD selects a recording
        inside its HDF5 file; camera datasets use one file per recording.

    Returns
    -------
    numpy.ndarray
        Events with columns ``x, y, polarity, t_ms``. Auditory channels use
        ``x=channel, y=0, polarity=0``; seconds are converted to milliseconds.

    Raises
    ------
    ValueError
        For an unsupported dataset, a path outside the root, malformed
        binary records, or incompatible sample indices and event columns.
    OSError
        If a verified recording is no longer readable.

    Notes
    -----
    No download or synthetic substitution occurs. The caller must bind the
    sample metadata to an actual file manifest before invoking this reader.
    """
    path = (root / sample.file).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("event sample path lies outside the dataset root")
    if isinstance(sample.index, bool) or sample.index < 0:
        raise ValueError("event sample index must be non-negative")
    if dataset == "shd":
        events, label = read_shd_recording(path, sample.index)
        if label != sample.label:
            raise ValueError("auditory recording label does not match its manifest sample")
        return events
    if sample.index != 0:
        raise ValueError("camera recording files hold exactly one sample")
    if dataset == "nmnist":
        return _parse_nmnist_bin(path)
    if dataset == "dvs_cifar10":
        return _parse_dvs_npy(path)
    raise ValueError("unsupported event dataset")

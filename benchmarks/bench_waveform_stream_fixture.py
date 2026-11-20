# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recorded EEG fixture for waveform-stream measurements

"""Read the original digital values from the pinned 64-channel PhysioNet recording."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SOURCE = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
SHA256 = "4743b736131a7e147c150e8b37711029b6cda5e356c4b3e8261a03cdcaaf8b0c"
SAMPLE_RATE = 160


def read_recording(path: Path) -> np.ndarray[Any, Any]:
    """Read EDF+ S001R01 digital samples as 9,760 rows by 64 simultaneous electrodes."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError("expected the original PhysioNet EEGMMIDB 1.0.0 S001R01.edf file")
    # The pinned file has 61 one-second records. Each record stores 160 int16
    # samples per electrode, then 80 words of EDF+ annotations.
    records, electrodes, annotation_words, header_bytes = 61, 64, 80, 16896
    data = np.ndarray(
        (records, electrodes, SAMPLE_RATE),
        dtype="<i2",
        buffer=source,
        offset=header_bytes,
        strides=(2 * (electrodes * SAMPLE_RATE + annotation_words), 2 * SAMPLE_RATE, 2),
    )
    return np.ascontiguousarray(
        data.transpose(0, 2, 1).reshape(records * SAMPLE_RATE, electrodes), dtype=np.int16
    )

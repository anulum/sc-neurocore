# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verify publisher SHD bytes before declaring training input

"""Create a manifest for the verified SHD publisher files already held locally.

These digests identify one exact publisher snapshot, rather than accepting any
HDF5 file with the same name. No dataset, executable or dependency is downloaded.
The data licence is CC-BY-4.0. Publisher source and checksum list:
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
https://zenkelab.org/datasets/md5sums.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sc_neurocore.datasets.manifest import EventDatasetManifest, build_manifest

PUBLISHER_FILES = {
    "shd_train.h5": (
        268826085,
        "2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1",
    ),
    "shd_test.h5": (
        78719235,
        "47d8621a092cb483bea9448a93c0a3375a6726d370045090c2f7a77f70f95df3",
    ),
}


def verified_manifest(root: Path) -> EventDatasetManifest:
    """Verify the exact publisher snapshot and return its public data declaration.

    Parameters
    ----------
    root : Path
        Directory containing both uncompressed publisher HDF5 files.

    Returns
    -------
    EventDatasetManifest
        Manifest with the publisher file digests as the release identity.

    Raises
    ------
    ValueError
        File bytes differ from this snapshot.
    FileNotFoundError
        A required recording file is missing.
    """
    version = "sha256:" + "+".join(value[1] for value in PUBLISHER_FILES.values())
    manifest = build_manifest("shd", root, version=version)
    actual = {record.path: (record.bytes, record.sha256) for record in manifest.files}
    if actual != PUBLISHER_FILES:
        raise ValueError("SHD bytes do not match the declared publisher snapshot")
    return manifest


def main() -> None:
    """Write the verified manifest without overwriting an existing declaration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verified_manifest(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "manifest_digest": manifest.digest,
                "samples": dict(Counter(sample.split for sample in manifest.samples)),
            }
        )
    )


if __name__ == "__main__":
    main()

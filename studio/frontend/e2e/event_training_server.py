# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event browser acceptance server fixture

"""Serve real process training over disposable published-format event files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio import project
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.platform import StudioRuntimeSettings
from tests.event_dataset_support import write_nmnist


def main() -> None:
    """Create a fresh owned test corpus and run the unmodified Studio API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = Path(os.environ["SC_NEUROCORE_STUDIO_EVENT_TEST_ROOT"])
    if not root.is_absolute():
        raise ValueError("event browser test root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    recordings = root / "recordings"
    if recordings.exists():
        raise ValueError("event browser acceptance needs a fresh recording directory")
    write_nmnist(recordings, {"train": {0: 2, 1: 2}, "test": {0: 1}})
    manifest = build_manifest("nmnist", recordings, version="generated-browser-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7),
        EventBinning(1.002, 4, 34, 34, "separate"),
        "train",
        "evaluation",
    )
    (root / "event_data.json").write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    os.environ["SC_NEUROCORE_STUDIO_DATASET_ROOT"] = str(recordings)
    # Only the existing storage path is isolated; routes and workspace logic stay real.
    project._PROJECTS_DIR = str(root / "projects")
    app = create_app(StudioRuntimeSettings(job_root_path=str(root / "jobs")))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

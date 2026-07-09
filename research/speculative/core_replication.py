# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
import os
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VonNeumannProbe:
    """
    Simulates a self-replicating code entity.
    Can copy the sc-neurocore source and its own state to a new 'host'.
    """

    probe_id: int

    def replicate(self, destination_dir: str):  # pragma: no cover
        """
        Quine-like behavior: Copies the library source to a new location.
        Uses shutil.copytree which can fail on platform-specific special files.
        """
        # Path sanitization: resolve and reject path traversal
        destination_dir = os.path.realpath(destination_dir)
        if ".." in os.path.relpath(destination_dir, os.getcwd()):
            raise ValueError("Destination must be within or below the current working directory.")

        logger.info("Probe %d: Replicating to %s...", self.probe_id, destination_dir)

        # 1. Identify source root
        # (Assuming we are in src/sc_neurocore/core/replication.py)
        src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        try:
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
        except OSError as exc:
            logger.error("Probe %d: Failed to create destination: %s", self.probe_id, exc)
            raise

        # 2. Copy source files
        # Only copy the library 'sc_neurocore' folder
        lib_src = os.path.join(src_root, "sc_neurocore")
        lib_dst = os.path.join(destination_dir, "sc_neurocore")

        if os.path.exists(lib_dst):
            shutil.rmtree(lib_dst)

        shutil.copytree(lib_src, lib_dst)

        # 3. Create a launch script in the new destination
        launch_script = os.path.join(destination_dir, "launch_probe.py")
        with open(launch_script, "w") as f:
            f.write("import sys\nimport os\n")
            f.write("sys.path.append(os.path.abspath('.'))\n")
            f.write("from sc_neurocore.core.replication import VonNeumannProbe\n")
            f.write(f"p = VonNeumannProbe(probe_id={self.probe_id + 1})\n")
            f.write("print('Probe sequence initiated in new sector.')\n")

        logger.info("Probe %d: Success. New generation ready.", self.probe_id)

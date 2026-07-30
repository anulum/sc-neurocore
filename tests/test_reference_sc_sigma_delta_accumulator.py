# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import hashlib
import json
import struct
from pathlib import Path
from sc_neurocore.neurons.models.sc_sigma_delta_accumulator import SCSigmaDeltaAccumulatorNeuron


def test_project_receipt_matches_frozen_recurrence() -> None:
    receipt = json.loads(
        (
            Path(__file__).parents[1]
            / "src/sc_neurocore/neurons/reference_trace_data/sc_sigma_delta_accumulator_project.json"
        ).read_text()
    )
    drive = [0.0] * 32 + [0.3] * 96 + [-0.7, 1.1] * 64
    n = SCSigmaDeltaAccumulatorNeuron()
    digest = hashlib.sha256()
    for current in drive:
        event = n.step(current)
        digest.update(struct.pack("<di", n.sigma, event))
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]

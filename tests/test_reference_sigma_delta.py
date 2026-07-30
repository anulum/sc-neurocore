# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import hashlib
import json
import math
import struct
from pathlib import Path
from sc_neurocore.neurons.models.sigma_delta import SigmaDeltaNeuron


def test_source_receipt_matches_independent_equations_and_production() -> None:
    receipt = json.loads(
        (
            Path(__file__).parents[1]
            / "src/sc_neurocore/neurons/reference_trace_data/sigma_delta_apsdm.json"
        ).read_text()
    )
    drive = [0.0] * 32 + [2.0] * 128 + [-1.0, 4.0] * 176
    sigma = reconstruction = 0.0
    events = []
    digest = hashlib.sha256()
    production = SigmaDeltaNeuron()
    for current in drive:
        sigma += 0.1 * current
        reconstruction *= math.exp(-0.1 / 10.0)
        event = int(sigma - reconstruction >= 0.5)
        if event:
            reconstruction += 1.0
        events.append(event)
        digest.update(struct.pack("<ddB", sigma, reconstruction, event))
        assert production.step(current) == event
        assert production.sigma == sigma
        assert production.reconstruction == reconstruction
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]
    assert sum(events) == 276

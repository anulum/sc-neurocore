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
from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron


def test_fardet_levina_receipt_matches_production() -> None:
    receipt = json.loads(
        (
            Path(__file__).parents[1]
            / "src/sc_neurocore/neurons/reference_receipts/energy_lif_fardet_levina.json"
        ).read_text()
    )
    n = EnergyLIFNeuron()
    digest = hashlib.sha256()
    events = 0
    for i in range(512):
        event = n.step((80.0, 0.0, 120.0, 20.0)[i % 4])
        events += event
        digest.update(struct.pack("<ddB", n.v, n.epsilon, event))
    assert events == receipt["oracle"]["events"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]

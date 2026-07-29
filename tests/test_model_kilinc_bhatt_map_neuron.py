# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — compatibility identity contract

from sc_neurocore.neurons.models import KilincBhattMapNeuron
from sc_neurocore.neurons.models import SCAdaptiveThresholdMapNeuron
from sc_neurocore.neurons.models.kilinc_bhatt_map_neuron import (
    KilincBhattMapNeuron as DirectAlias,
)


def test_legacy_name_is_only_an_alias_to_the_retained_project_model() -> None:
    assert KilincBhattMapNeuron is SCAdaptiveThresholdMapNeuron is DirectAlias
    neuron = KilincBhattMapNeuron()
    assert neuron.step(0.6) == 1
    assert (neuron.x, neuron.theta) == (1.35, 0.0)

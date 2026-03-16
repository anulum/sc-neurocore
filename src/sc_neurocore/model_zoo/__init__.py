# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model Zoo: pre-configured network-level SNN architectures

"""Pre-configured network-level SNN architectures.

Load and run a model in three lines::

    from sc_neurocore.model_zoo import brunel_balanced_network
    net = brunel_balanced_network()
    net.run(0.1)
"""

from __future__ import annotations

from .configs import (
    mnist_classifier,
    dvs_gesture_classifier,
    shd_speech_classifier,
    brunel_balanced_network,
    cortical_column,
    central_pattern_generator,
    decision_making_circuit,
    working_memory_circuit,
    auditory_processing,
    visual_cortex_v1,
)

__all__ = [
    "mnist_classifier",
    "dvs_gesture_classifier",
    "shd_speech_classifier",
    "brunel_balanced_network",
    "cortical_column",
    "central_pattern_generator",
    "decision_making_circuit",
    "working_memory_circuit",
    "auditory_processing",
    "visual_cortex_v1",
]

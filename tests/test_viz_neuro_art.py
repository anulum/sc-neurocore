# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import numpy as np

from sc_neurocore.viz.neuro_art import NeuroArtGenerator


def test_neuro_art_generation_is_deterministic_for_identical_neural_state():
    generator = NeuroArtGenerator(resolution=32)
    state = np.array([0.2, -0.4, 0.8, 0.1])

    first = generator.generate_visual(state)
    second = generator.generate_visual(state.copy())

    assert first.shape == (32, 32, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) > 0


def test_neuro_art_empty_state_returns_blank_canvas_with_requested_resolution():
    image = NeuroArtGenerator(resolution=16).generate_visual(np.array([]))

    assert image.shape == (16, 16, 3)
    assert image.dtype == np.uint8
    assert np.count_nonzero(image) == 0

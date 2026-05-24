# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for QEC shield contracts

"""Contracts for non-repetition QEC shield passthrough modes."""

from __future__ import annotations

import numpy as np

from sc_neurocore.quantum.qec import QecShield


def test_surface_qec_passthrough_preserves_payload_identity_and_shape() -> None:
    qec = QecShield(code_type="surface", distance=3)
    bits = np.random.randint(0, 2, (4, 3, 64), dtype=np.uint8)

    assert qec.encode(bits) is bits
    assert qec.extract_syndromes(bits).shape == bits.shape
    assert qec.decode(bits) is bits

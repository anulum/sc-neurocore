# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstruction from former test_arcane_zenith.py

"""Focused suite: TestConstruction from former test_arcane_zenith.py."""

from __future__ import annotations

import pytest
from sc_neurocore.arcane_zenith import (
    ArcaneZenithCognitiveCore,
    create_arcane_neuron_with_zenith_plasticity,
)
from sc_neurocore.neurons.models.arcane_neuron import (
    ArcaneNeuron,
)

pytest.importorskip("torch")


class TestConstruction:
    """Construction of the Arcane-Zenith cognitive core."""

    def test_factory_returns_instance(self) -> None:
        """The factory returns a cognitive core."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        assert isinstance(core, ArcaneZenithCognitiveCore)

    def test_direct_init_wires_arcane_neuron(self) -> None:
        """Direct construction wires an Arcane neuron."""
        core = ArcaneZenithCognitiveCore(backend="torch")
        assert isinstance(core.neuron, ArcaneNeuron)

    def test_four_independent_plasticity_rules(self) -> None:
        """The core holds four distinct plasticity rules."""
        core = ArcaneZenithCognitiveCore(backend="torch")
        rules = (core.tau_rule, core.nov_rule, core.conf_rule, core.lr_rule)
        # Same concrete class but four distinct object identities.
        assert len({id(r) for r in rules}) == 4

    def test_initial_weights_follow_design(self) -> None:
        """The four rules start at their designed weights."""
        core = ArcaneZenithCognitiveCore(backend="torch")
        # Documented initialisation in arcane_zenith.py:
        # tau=0.5, nov=0.2, conf=0.3, lr=0.1
        assert float(core.tau_rule.get_weights()[0]) == pytest.approx(0.5, abs=1e-6)
        assert float(core.nov_rule.get_weights()[0]) == pytest.approx(0.2, abs=1e-6)
        assert float(core.conf_rule.get_weights()[0]) == pytest.approx(0.3, abs=1e-6)
        assert float(core.lr_rule.get_weights()[0]) == pytest.approx(0.1, abs=1e-6)

    def test_unknown_backend_rejected(self) -> None:
        """An unknown backend is refused."""
        with pytest.raises(ValueError, match="unknown backend"):
            ArcaneZenithCognitiveCore(backend="not-a-backend")

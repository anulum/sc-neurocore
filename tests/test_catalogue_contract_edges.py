# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Small contract edges of the catalogue, lanes and phase layer

"""Answers the catalogue gives at its edges: absent files, stochastic triggers, scalars."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons import model_catalogue
from sc_neurocore.neurons.descriptor_generator import _instance_state_is_vector
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models import AdExNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.runtime_lanes import _importable
from sc_neurocore.scpn.layers.l8_phase_field import L8_PhaseFieldLayer, L8_StochasticParameters


class NeedsAnArgument:
    """A model class that cannot be built with no arguments."""

    def __init__(self, size: int) -> None:
        self.state = [0.0] * size


def test_a_state_field_is_a_vector_only_when_a_default_instance_holds_one() -> None:
    assert _instance_state_is_vector(AdExNeuron, "v") is False
    assert _instance_state_is_vector(NeedsAnArgument, "state") is False


@pytest.mark.parametrize(
    ("neuron", "noisy"),
    [
        (EquationNeuron({"v": "-v"}, detection="escape_rate", rate_expression="0.5*v"), False),
        (
            EquationNeuron({"v": "-v"}, detection="poisson", probability_expression="0.1*xi"),
            True,
        ),
    ],
    ids=["escape-rate", "poisson"],
)
def test_diffusion_noise_is_found_in_a_stochastic_trigger_expression(
    neuron: EquationNeuron, noisy: bool
) -> None:
    assert neuron.uses_diffusion_noise is noisy


def test_a_registered_descriptor_missing_from_disk_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every shipped descriptor is on disk, so a partial install is produced by registering one that is not."""
    monkeypatch.setitem(model_catalogue._DESCRIPTOR_PATHS, "GhostNeuron", tmp_path / "Ghost.toml")
    assert model_catalogue.load_descriptor_payload("GhostNeuron") is None


def test_a_schema_neuron_reports_the_profile_it_was_authored_with() -> None:
    neuron = UniversalNeuron.from_schema("lif")
    assert neuron.profile is neuron._profile
    assert neuron.profile.stem == "LIF"


def test_a_lane_module_is_probed_without_importing_it() -> None:
    assert _importable("sc_neurocore_no_such_lane_module") is False
    assert _importable("this") is True


def test_a_phase_layer_without_pulsar_frequencies_refuses_to_step() -> None:
    layer = L8_PhaseFieldLayer(L8_StochasticParameters(n_pulsars=2, bitstream_length=8, rng_seed=1))
    layer.params.pulsar_omegas = None  # simulate corrupted layer state
    with pytest.raises(RuntimeError, match="pulsar frequencies were not initialised"):
        layer.step(0.001)

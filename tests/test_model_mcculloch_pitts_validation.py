# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts validation and transport contracts

"""Constructor, step, transport, and mutation validation contracts."""

from .model_mcculloch_pitts_support import *


@pytest.mark.parametrize(
    "theta",
    (0, -1, 1.5, True, np.nan, np.inf, -np.inf, _INT32_MAX + 1, "1"),
)
def test_constructor_rejects_non_positive_or_lossy_thresholds(theta: object) -> None:
    """The fixed threshold is a positive signed-ABI-safe afferent count."""
    with pytest.raises(ValueError, match="theta"):
        McCullochPittsNeuron(theta=cast(int, theta))


@pytest.mark.parametrize(
    "count",
    (-1, 0.5, True, np.nan, np.inf, -np.inf, _INT32_MAX + 1, "1", object()),
)
def test_step_rejects_invalid_excitatory_counts(count: object) -> None:
    """Negative, fractional, Boolean and out-of-domain counts fail closed."""
    with pytest.raises(ValueError, match="excitatory_count"):
        McCullochPittsNeuron().step(count)


@pytest.mark.parametrize("flag", (0, 1, 0.0, 1.0, "false", None))
def test_step_requires_an_exact_inhibitory_boolean(flag: object) -> None:
    """Numeric truthiness cannot silently alter the absolute-veto wire."""
    with pytest.raises(ValueError, match="inhibitory_active"):
        McCullochPittsNeuron().step(1, flag)


def test_integer_valued_transport_floats_and_numpy_scalars_are_normalised() -> None:
    """Network Float64 transport remains usable without accepting fractions."""
    neuron = McCullochPittsNeuron(theta=cast(int, np.float64(2.0)))
    assert type(neuron.theta) is int
    assert neuron.step(np.float64(2.0), np.bool_(False)) == 1
    assert neuron.step(np.int32(1), np.bool_(False)) == 0


def test_runtime_threshold_corruption_fails_closed() -> None:
    """Public dataclass mutation is revalidated on step and reset."""
    neuron = McCullochPittsNeuron(theta=2)
    neuron.theta = cast(int, 1.25)
    with pytest.raises(ValueError, match="theta"):
        neuron.step(2)
    with pytest.raises(ValueError, match="theta"):
        neuron.reset()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thermal SC probability encoding contracts

"""Single-probability and configuration contracts for thermal SC encoding."""

from typing import Any, cast
from sc_neurocore.security.thermal_sc_encoding import ThermalSCEncodingConfig
from sc_neurocore.security.thermal_sc_encoding import ThermalSCEncodingError
from sc_neurocore.security.thermal_sc_encoding import encode_activity_balanced_probability
import pytest


def test_activity_balanced_encoder_preserves_probability_and_balances_switching() -> None:
    """A probability of 0.5 keeps four ones in eight and switches at every step."""
    result = encode_activity_balanced_probability(
        0.5,
        ThermalSCEncodingConfig(bitstream_length=8, seed=11),
        stream_index=0,
    )

    assert result.bitstream.count(1) == 4
    assert result.realised_probability == 0.5
    assert result.activity_summary.per_stream_transition_counts == (7,)
    assert result.activity_summary.mean_transition_rate == 1.0
    assert result.evidence_boundary == "analytic_simulation_only"


def test_seed_domain_rotation_changes_phase_without_changing_activity_or_value() -> None:
    """A new stream index rotates the stream but keeps its ones, activity and value."""
    config = ThermalSCEncodingConfig(bitstream_length=16, seed=7, rotation_stride=3)

    first = encode_activity_balanced_probability(0.25, config, stream_index=0)
    second = encode_activity_balanced_probability(0.25, config, stream_index=1)

    assert first.bitstream != second.bitstream
    assert first.bitstream.count(1) == second.bitstream.count(1) == 4
    assert first.activity_summary == second.activity_summary
    assert first.seed_domain != second.seed_domain


@pytest.mark.parametrize(
    ("probability", "config"),
    [
        (-0.1, ThermalSCEncodingConfig(bitstream_length=8)),
        (1.1, ThermalSCEncodingConfig(bitstream_length=8)),
        (0.5, ThermalSCEncodingConfig(bitstream_length=1)),
        (0.5, ThermalSCEncodingConfig(bitstream_length=8, rotation_stride=0)),
        (
            0.5,
            ThermalSCEncodingConfig(
                bitstream_length=8,
                dummy_streams_per_record=1,
                max_dummy_overhead_ratio=0.5,
            ),
        ),
    ],
)
def test_activity_balanced_encoder_rejects_invalid_contracts(
    probability: float,
    config: ThermalSCEncodingConfig,
) -> None:
    """An invalid probability or configuration is rejected."""
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probability(probability, config)


def test_activity_balanced_encoder_rejects_boolean_stream_index() -> None:
    """A boolean stream index is rejected."""
    with pytest.raises(ThermalSCEncodingError, match="stream_index must be a non-negative integer"):
        encode_activity_balanced_probability(
            0.5,
            ThermalSCEncodingConfig(bitstream_length=8),
            stream_index=True,
        )


def test_validate_config_rejects_non_config_object() -> None:
    """A config of the wrong type is rejected before any field is read."""
    with pytest.raises(ThermalSCEncodingError, match="must be a ThermalSCEncodingConfig"):
        encode_activity_balanced_probability(0.5, cast(Any, object()))


@pytest.mark.parametrize(
    "config",
    [
        ThermalSCEncodingConfig(bitstream_length=8, seed=-1),
        ThermalSCEncodingConfig(bitstream_length=8, dummy_streams_per_record=-1),
        ThermalSCEncodingConfig(bitstream_length=8, max_dummy_overhead_ratio=-1.0),
        ThermalSCEncodingConfig(bitstream_length=8, max_dummy_overhead_ratio=float("inf")),
    ],
)
def test_validate_config_rejects_out_of_range_fields(config: ThermalSCEncodingConfig) -> None:
    """Out-of-range configuration fields are rejected.

    A negative seed, negative dummy-stream count, and a negative or
    non-finite overhead ratio are each rejected as contract violations.
    """
    with pytest.raises(ThermalSCEncodingError):
        encode_activity_balanced_probability(0.5, config)


def test_validate_probability_rejects_non_numeric() -> None:
    """A non-numeric probability fails the type guard before the range check."""
    with pytest.raises(ThermalSCEncodingError, match=r"probability must be a finite value"):
        encode_activity_balanced_probability(
            cast(Any, "oops"),
            ThermalSCEncodingConfig(bitstream_length=8),
        )


@pytest.mark.parametrize(("probability", "expected_ones"), [(0.0, 0), (1.0, 8)])
def test_distribute_ones_handles_degenerate_densities(
    probability: float, expected_ones: int
) -> None:
    """Densities of zero and one give zero and eight ones.

    All-zero and all-one densities take the dedicated fast paths instead of
    the interior rank-spreading loop.
    """
    result = encode_activity_balanced_probability(
        probability, ThermalSCEncodingConfig(bitstream_length=8)
    )
    assert result.bitstream.count(1) == expected_ones
    assert result.realised_probability == probability

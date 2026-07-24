# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynapsePrecision from former test_adaptive_precision.py

"""Focused suite: TestSynapsePrecision from former test_adaptive_precision.py."""

from __future__ import annotations

from tests.adaptive_precision_support import *  # noqa: F403


class TestSynapsePrecision:
    """Per-synapse precision-planning checks."""

    def test_synapse_precision_to_dict_is_stable(self) -> None:
        """SynapsePrecision serializes in manifest field order."""
        row = SynapsePrecision(
            layer_index=0,
            layer_name="fc",
            output_index=1,
            input_index=2,
            bit_width=8,
            bitstream_length=128,
            sensitivity=0.5,
            quantization_error_bound=0.01,
            stochastic_error_bound=0.02,
            total_error_bound=0.03,
        )

        assert row.to_dict() == {
            "layer_index": 0,
            "layer_name": "fc",
            "output_index": 1,
            "input_index": 2,
            "bit_width": 8,
            "bitstream_length": 128,
            "sensitivity": 0.5,
            "quantization_error_bound": 0.01,
            "stochastic_error_bound": 0.02,
            "total_error_bound": 0.03,
        }

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (
                lambda: SynapsePrecision(-1, "fc", 0, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "layer_index",
            ),
            (
                lambda: SynapsePrecision(0, "", 0, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "layer_name",
            ),
            (
                lambda: SynapsePrecision(0, "fc", -1, 1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "output_index",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, -1, 8, 128, 0.5, 0.01, 0.02, 0.03),
                "input_index",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 0, 128, 0.5, 0.01, 0.02, 0.03),
                "bit_width",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 0, 0.5, 0.01, 0.02, 0.03),
                "bitstream_length",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, -0.1, 0.01, 0.02, 0.03),
                "sensitivity",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, -0.1, 0.02, 0.03),
                "quantization_error_bound",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, 0.01, -0.1, 0.03),
                "stochastic_error_bound",
            ),
            (
                lambda: SynapsePrecision(0, "fc", 0, 1, 8, 128, 0.5, 0.01, 0.02, -0.1),
                "total_error_bound",
            ),
        ],
    )
    def test_rejects_invalid_synapse_precision_fields(
        self,
        factory: Callable[[], SynapsePrecision],
        message: str,
    ) -> None:
        """SynapsePrecision rejects impossible per-synapse rows."""
        with pytest.raises(ValueError, match=message):
            factory()

    def test_rejects_total_error_bound_below_components(self) -> None:
        """Total error must include quantization and stochastic components."""
        with pytest.raises(ValueError, match="total_error_bound"):
            SynapsePrecision(
                layer_index=0,
                layer_name="fc",
                output_index=0,
                input_index=1,
                bit_width=8,
                bitstream_length=128,
                sensitivity=0.5,
                quantization_error_bound=0.02,
                stochastic_error_bound=0.02,
                total_error_bound=0.03,
            )

    def test_rejects_non_numeric_synapse_sensitivity(self) -> None:
        """SynapsePrecision rejects non-numeric sensitivity payloads at runtime."""
        with pytest.raises(ValueError, match="sensitivity"):
            SynapsePrecision(
                layer_index=0,
                layer_name="fc",
                output_index=0,
                input_index=1,
                bit_width=8,
                bitstream_length=128,
                sensitivity=cast(float, "bad"),
                quantization_error_bound=0.01,
                stochastic_error_bound=0.02,
                total_error_bound=0.03,
            )

    def test_assign_synapse_precisions_returns_one_row_per_weight(self) -> None:
        """Per-synapse planning returns one assignment per weight element."""
        weights = [np.array([[0.1, 0.8], [0.0, 0.4]])]

        result = assign_synapse_precisions(
            weights,
            layer_names=["fc"],
            target_error=0.05,
            min_bits=3,
            max_bits=8,
            min_length=16,
            max_length=256,
        )

        assert len(result) == 4
        assert all(isinstance(row, SynapsePrecision) for row in result)
        assert {row.layer_name for row in result} == {"fc"}
        assert all(3 <= row.bit_width <= 8 for row in result)
        assert all(16 <= row.bitstream_length <= 256 for row in result)
        assert all(row.total_error_bound >= row.quantization_error_bound for row in result)

    def test_high_sensitivity_gets_at_least_as_much_precision(self) -> None:
        """Higher-sensitivity weights receive at least as much precision."""
        weights = [np.array([[0.01, 1.0]])]
        result = assign_synapse_precisions(
            weights,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        low, high = result
        assert high.sensitivity > low.sensitivity
        assert high.bit_width >= low.bit_width
        assert high.bitstream_length >= low.bitstream_length

    def test_custom_sensitivity_map_controls_precision(self) -> None:
        """Custom sensitivity maps influence per-synapse precision."""
        weights = [np.array([[0.5, 0.5]])]
        sensitivities = [np.array([[0.1, 1.0]])]

        result = assign_synapse_precisions(
            weights,
            sensitivity_maps=sensitivities,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        assert result[1].sensitivity > result[0].sensitivity
        assert result[1].bit_width >= result[0].bit_width

    def test_precision_plan_manifest_is_deterministic(self) -> None:
        """Precision plan manifests expose deterministic keys and costs."""
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.1,
            min_bits=2,
            max_bits=8,
        )

        manifest = precision_plan_manifest(assignments)

        assert manifest["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert manifest["granularity"] == "synapse"
        assert manifest["num_synapses"] == 2
        assert manifest["max_total_error_bound"] >= 0
        assert manifest["cost_summary"]["estimated_lut_cost"] > 0.0
        assert (
            manifest["cost_summary"]["uniform_length_reference_cost"]
            >= manifest["cost_summary"]["estimated_lut_cost"]
        )
        assert manifest["cost_summary"]["estimated_lut_savings_vs_uniform_length"] >= 0.0
        assert list(manifest["assignments"][0]) == [
            "layer_index",
            "layer_name",
            "output_index",
            "input_index",
            "bit_width",
            "bitstream_length",
            "sensitivity",
            "quantization_error_bound",
            "stochastic_error_bound",
            "total_error_bound",
        ]

    def test_rejects_invalid_sensitivity_shape(self) -> None:
        """Sensitivity maps must match their weight tensor shapes."""
        with np.testing.assert_raises_regex(ValueError, "sensitivity map"):
            assign_synapse_precisions(
                [np.array([[0.5, 0.5]])],
                sensitivity_maps=[np.array([[0.1], [1.0]])],
            )

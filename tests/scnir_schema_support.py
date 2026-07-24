# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_scnir_schema.py

from __future__ import annotations


"""Contract tests for the stochastic-computing NIR metadata layer."""


import json


from pathlib import Path


from unittest import mock


import pytest


from sc_neurocore.cli import main


from sc_neurocore.ir.scnir_schema import (
    SCNIR_PREVIOUS_SCHEMA_VERSION,
    SCNIR_SCHEMA_VERSION,
    SCNIR_V02_SCHEMA_VERSION,
    SCNIRCorrelationConstraint,
    SCNIRDocument,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    SCNIRStreamTransform,
    SCNIRValidationError,
    load_scnir,
    scnir_from_dict,
    scnir_to_dict,
    upgrade_scnir_dict,
    validate_scnir_dict,
    write_scnir,
)


from sc_neurocore.learning.online_o1 import OnlineO1Config


def _valid_document() -> SCNIRDocument:
    return SCNIRDocument(
        producer="sc-neurocore-test",
        streams=[
            SCNIRStream(
                stream_id="layer0_input",
                layer="layer0",
                bitstream_length=1024,
                encoding="bipolar",
                signal_kind="spike",
                precision=SCNIRPrecision(
                    signed=True,
                    total_bits=16,
                    fractional_bits=8,
                    accumulator_bits=32,
                    rounding="nearest_even",
                    overflow="saturate",
                ),
                source=SCNIRSource(
                    kind="lfsr",
                    seed=17,
                    lfsr_polynomial="x^16 + x^14 + x^13 + x^11 + 1",
                    tap_mask=0xB400,
                ),
                correlation_constraints=[
                    SCNIRCorrelationConstraint(
                        peer_stream_id="layer0_weight",
                        policy="max_correlation",
                        max_abs_correlation=0.03,
                    )
                ],
            ),
            SCNIRStream(
                stream_id="layer0_weight",
                layer="layer0",
                bitstream_length=1024,
                encoding="unipolar",
                signal_kind="weight",
                precision=SCNIRPrecision(
                    signed=False,
                    total_bits=12,
                    fractional_bits=10,
                    accumulator_bits=24,
                    rounding="stochastic",
                    overflow="error",
                ),
                source=SCNIRSource(kind="sobol", sobol_dimension=3),
            ),
        ],
    )


__all__ = ['json', 'Path', 'mock', 'pytest', 'main', 'SCNIR_PREVIOUS_SCHEMA_VERSION', 'SCNIR_SCHEMA_VERSION', 'SCNIR_V02_SCHEMA_VERSION', 'SCNIRCorrelationConstraint', 'SCNIRDocument', 'SCNIRHierarchyInstance', 'SCNIRHierarchyPort', 'SCNIRPrecision', 'SCNIRSource', 'SCNIRStream', 'SCNIRStreamTransform', 'SCNIRValidationError', 'load_scnir', 'scnir_from_dict', 'scnir_to_dict', 'upgrade_scnir_dict', 'validate_scnir_dict', 'write_scnir', 'OnlineO1Config', '_valid_document']


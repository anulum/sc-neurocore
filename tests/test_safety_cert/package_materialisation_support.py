# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_package_materialisation.py

from __future__ import annotations

"""Exercise fail-closed assembly, hashing, atomic writes, and verification."""

import hashlib


import json


import stat


from dataclasses import replace


from pathlib import Path


from typing import Any


import pytest


import sc_neurocore.safety_cert.certification as certification_module


from sc_neurocore.safety_cert import (
    CertificationGenerator,
    CertificationPackage,
    ChecklistItem,
    EvidenceBag,
    EvidenceItem,
    FailureCategory,
    FailureMode,
    FormalProofCertificate,
    FormalProperty,
    SILLevel,
    SafetyManualGenerator,
    SafetyStandard,
)


_GENERATED_AT = "2026-07-12T18:30:00+00:00"


_NETWORK_CONFIG = {
    "bitstream_length": 256,
    "num_inputs": 8,
    "num_neurons": 16,
    "clock_mhz": 100.0,
}


def _unsafe(value: object) -> Any:
    """Return an invalid runtime value for a deliberate boundary test."""
    return value


def _property() -> FormalProperty:
    return FormalProperty(
        prop_id="P-SAFE-001",
        module="neuron",
        description="Accumulator remains in range",
        property_type="assert",
        status="proven",
        engine="SymbiYosys 2.4.0",
        depth=32,
        sby_file="formal/neuron.sby",
    )


def _package(*, explicit_evidence: bool = True) -> CertificationPackage:
    generator = CertificationGenerator()
    if not explicit_evidence:
        return generator.generate(
            SafetyStandard.IEC_61508,
            SILLevel.SIL_2,
            ["neuron"],
            [_property()],
            generated_at=_GENERATED_AT,
        )
    return generator.generate(
        SafetyStandard.IEC_61508,
        SILLevel.SIL_2,
        ["neuron"],
        [_property()],
        _NETWORK_CONFIG,
        implementation_evidence={"neuron": ["rtl/neuron.sv"]},
        failure_modes=[
            FailureMode(
                "FM-NEURON-001",
                "neuron",
                "Accumulator register upset",
                FailureCategory.DANGEROUS_DETECTED,
                12.5,
                0.95,
                "Parity monitor; verification report E-17",
            )
        ],
        checklist_evidence={"7.4.2": "evidence/formal-review.md"},
        generated_at=_GENERATED_AT,
    )



__all__ = ['hashlib', 'json', 'stat', 'replace', 'Path', 'Any', 'pytest', 'certification_module', 'CertificationGenerator', 'CertificationPackage', 'ChecklistItem', 'EvidenceBag', 'EvidenceItem', 'FailureCategory', 'FailureMode', 'FormalProofCertificate', 'FormalProperty', 'SILLevel', 'SafetyManualGenerator', 'SafetyStandard', '_GENERATED_AT', '_NETWORK_CONFIG', '_unsafe', '_property', '_package']

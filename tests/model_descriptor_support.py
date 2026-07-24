# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_descriptor.py

from __future__ import annotations


"""Tests for the declarative model descriptor (schema v2) and its generator."""


import dataclasses


import importlib


import inspect


from pathlib import Path


import sys


import types


from typing import Any, Callable


import pytest


from sc_neurocore.neurons import universal_dsl


from sc_neurocore.neurons.descriptor_generator import (
    generate_descriptor,
    generate_descriptor_payload,
    merge_descriptor_payloads,
)


from sc_neurocore.neurons.model_descriptor import (
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptorError,
    Silicon,
    Validation,
    descriptor_completeness_tier,
    parse_model_descriptor,
)


from sc_neurocore.neurons.models import _CLASS_TO_MODULE


def _minimal_payload() -> dict[str, Any]:
    return {
        "metadata": {
            "schema_version": 2,
            "name": "AdEx",
            "class_name": "AdExNeuron",
            "module": "adex",
        },
        "state": {"v": {"init": -65.0}},
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
    }


__all__ = [
    "dataclasses",
    "importlib",
    "inspect",
    "Path",
    "sys",
    "types",
    "Any",
    "Callable",
    "pytest",
    "universal_dsl",
    "generate_descriptor",
    "generate_descriptor_payload",
    "merge_descriptor_payloads",
    "MODEL_DESCRIPTOR_SCHEMA_VERSION",
    "ModelDescriptorError",
    "Silicon",
    "Validation",
    "descriptor_completeness_tier",
    "parse_model_descriptor",
    "_CLASS_TO_MODULE",
    "_minimal_payload",
]

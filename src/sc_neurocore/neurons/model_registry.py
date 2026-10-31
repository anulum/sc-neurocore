# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public read-only model discovery

"""Discover catalogue identities without importing their implementations.

``MODEL_MODULES`` maps each registered Python class name to its module basename
under ``sc_neurocore.neurons.models``. It preserves the registry's declaration
order and includes source, project-original and SC identities. Historical
import-only aliases are excluded, just as they are from the catalogue.

The mapping is a read-only view of the same registry used by lazy class imports,
not a second inventory. It neither loads model classes nor attests scientific,
numerical or hardware readiness. Consumers must obtain those claims from the
descriptor, identity, profile and evidence APIs.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from sc_neurocore.neurons.models import _CLASS_TO_MODULE

#: Registered class identifiers and their implementation module basenames.
MODEL_MODULES: Final[Mapping[str, str]] = MappingProxyType(_CLASS_TO_MODULE)

__all__ = ["MODEL_MODULES"]

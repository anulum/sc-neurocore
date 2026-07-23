# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_schema_validator.py

from __future__ import annotations

"""Test suite for schema_validator.py — ensures all bundled schemas pass validation."""
from sc_neurocore.neurons.schema_validator import (
    validate_all_bundled,
    validate_schema,
    validate_schema_dict,
)

__all__ = ['validate_all_bundled', 'validate_schema', 'validate_schema_dict']

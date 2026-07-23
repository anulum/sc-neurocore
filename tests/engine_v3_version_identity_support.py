# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_engine_v3_version_identity.py

from __future__ import annotations

"""Version-identity checks shared by the v3 engine release phases."""
from importlib.metadata import PackageNotFoundError, version
import sc_neurocore
import sc_neurocore_engine as v3
def _assert_engine_version_matches_core() -> None:
    """Validate source-tree and installed-wheel version surfaces."""

    assert v3.__version__ == sc_neurocore.__version__
    try:
        installed_version = version("sc-neurocore-engine")
    except PackageNotFoundError:
        return
    assert installed_version == sc_neurocore.__version__

__all__ = ['PackageNotFoundError', 'version', 'sc_neurocore', 'v3', '_assert_engine_version_matches_core']

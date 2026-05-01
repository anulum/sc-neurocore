# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Hardware platform profiles for compiler target selection."""

from __future__ import annotations

from . import cmos_profiles as _cmos_profiles
from . import exotic_profiles as _exotic_profiles
from . import neuromorphic_profiles as _neuromorphic_profiles
from .registry import (
    _PROFILES,
    HardwareProfile,
    discover_platforms,
    get_profile,
    list_profile_names,
    list_profiles,
    load_profiles_from_toml,
    load_toml_profile,
    load_toml_profiles_dir,
    register_platform_hook,
)

_PROFILE_MODULES = (_cmos_profiles, _exotic_profiles, _neuromorphic_profiles)

__all__ = [
    "HardwareProfile",
    "_PROFILES",
    "discover_platforms",
    "get_profile",
    "list_profile_names",
    "list_profiles",
    "load_profiles_from_toml",
    "load_toml_profile",
    "load_toml_profiles_dir",
    "register_platform_hook",
]

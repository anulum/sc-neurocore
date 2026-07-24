# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_isolation.py

"""Module-level tests from former test_isolation.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from isolation_support import *  # noqa: F403


def test_violation_log_capacity_is_bounded() -> None:
    firewall = BitstreamFirewall()
    firewall.max_violations = 1

    assert firewall.check_access("tenant", 0x1000) is False
    assert firewall.check_access("tenant", 0x2000) is False
    assert firewall.violation_count == 1


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.FirewallRule is isolation.FirewallRule
    assert compatibility_surface.BitstreamFirewall is isolation.BitstreamFirewall
    assert compatibility_surface.verify_isolation is isolation.verify_isolation


def test_address_isolation_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(isolation.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}

    assert facade_classes.isdisjoint({"FirewallRule", "BitstreamFirewall"})
    assert "verify_isolation" not in facade_functions
    assert owner_classes == {"FirewallRule", "BitstreamFirewall"}
    assert owner_functions == {"verify_isolation"}
    assert len(Path(isolation.__file__).read_text(encoding="utf-8").splitlines()) <= 120
    assert len(Path(compatibility_surface.__file__).read_text(encoding="utf-8").splitlines()) <= 900

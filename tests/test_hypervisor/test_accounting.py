# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor resource-accounting contracts

"""Verify tenant usage isolation, invoice calculation, and definition ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sc_neurocore.hypervisor import accounting as accounting_owner
from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor.accounting import ResourceAccounting


class TestResourceAccounting:
    def test_record_and_query(self) -> None:
        ra = ResourceAccounting()
        ra.record("t0", 10000, 500)
        ra.record("t0", 5000, 300)
        assert ra.total_cycles("t0") == 15000
        assert ra.total_spikes("t0") == 800

    def test_invoice(self) -> None:
        ra = ResourceAccounting()
        ra.record("t0", 1_000_000, 0)
        assert ra.invoice("t0", cost_per_cycle=1e-6) == pytest.approx(1.0)

    def test_unknown_tenant(self) -> None:
        ra = ResourceAccounting()
        assert ra.total_cycles("nobody") == 0


def test_tenant_totals_are_isolated() -> None:
    accounting = ResourceAccounting()
    accounting.record("a", 10, 2)
    accounting.record("b", 20, 3)

    assert accounting.total_cycles("a") == 10
    assert accounting.total_spikes("a") == 2
    assert accounting.total_cycles("b") == 20
    assert accounting.total_spikes("b") == 3


def test_unknown_spikes_and_default_invoice_are_zero() -> None:
    accounting = ResourceAccounting()

    assert accounting.total_spikes("unknown") == 0
    assert accounting.invoice("unknown") == 0.0


def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.UsageRecord is accounting_owner.UsageRecord
    assert compatibility_surface.ResourceAccounting is accounting_owner.ResourceAccounting


def test_accounting_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(accounting_owner.__file__).read_text(encoding="utf-8"))

    facade_classes = {node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)}
    owner_classes = {node.name for node in owner_tree.body if isinstance(node, ast.ClassDef)}
    owned_names = {"UsageRecord", "ResourceAccounting"}

    assert facade_classes.isdisjoint(owned_names)
    assert owner_classes == owned_names

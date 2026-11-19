# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor capacity, bandwidth metering and priority scheduling

"""Hold placement to region capacity, metering to its units, and the priority plan to every tenant.

A region fits a tenant only when it has the neurons and the synapses the tenant's QoS asks for, for allocation and
migration alike. The bandwidth meter's rate counts the spikes after its window's first sample over the cycles they
span, and converts to Mbps only with the bits per spike and the clock the caller states. Several realtime tenants
share one half of the plan, so none of them receives nothing.
"""

from __future__ import annotations

import math

import pytest

from sc_neurocore.hypervisor.hypervisor import Hypervisor
from sc_neurocore.hypervisor.isolation import BitstreamFirewall, FirewallRule
from sc_neurocore.hypervisor.migration import MigrationEngine
from sc_neurocore.hypervisor.qos_monitor import BandwidthMeter
from sc_neurocore.hypervisor.region import HWRegion, RegionState
from sc_neurocore.hypervisor.scheduler import Scheduler, SchedulingPolicy
from sc_neurocore.hypervisor.tenant import QoSPolicy, Tenant, TenantPriority


def _region(rid: int, base: int, neurons: int = 1024, synapses: int = 16384) -> HWRegion:
    """Return a free region with the given capacity at an AXI base address."""
    return HWRegion(
        region_id=rid,
        num_neurons=neurons,
        num_synapses=synapses,
        axi_base_addr=base,
        axi_size=0x1000,
    )


def test_a_region_fits_a_tenant_only_with_enough_neurons_and_synapses() -> None:
    """Both quantities of the QoS bound placement."""
    qos = QoSPolicy(max_neurons=1024, max_synapses=16384)
    assert _region(0, 0, 1024, 16384).fits(qos) is True
    assert _region(0, 0, 1023, 16384).fits(qos) is False
    assert _region(0, 0, 1024, 16383).fits(qos) is False


def test_allocation_skips_a_region_without_enough_synapses() -> None:
    """A region with the neurons but too few synapses is not given to the tenant."""
    hypervisor = Hypervisor()
    hypervisor.add_region(_region(0, 0x4000_0000, synapses=100))
    hypervisor.add_region(_region(1, 0x4000_1000))
    hypervisor.register_tenant(Tenant(tenant_id="a", name="a"))
    assert hypervisor.allocate("a") == 1
    assert hypervisor.regions[0].is_free


@pytest.mark.parametrize(("neurons", "synapses"), [(512, 16384), (1024, 100)])
def test_migration_to_a_region_too_small_changes_nothing(neurons: int, synapses: int) -> None:
    """A target short of neurons or synapses is refused before any state moves."""
    engine, firewall = MigrationEngine(), BitstreamFirewall()
    source, target = _region(0, 0x4000_0000), _region(1, 0x4000_1000, neurons, synapses)
    source.state, source.tenant_id = RegionState.ALLOCATED, "a"
    tenant = Tenant(tenant_id="a", name="a", region_id=0, active=True)
    firewall.add_rule(FirewallRule("a", 0x4000_0000, 0x1000))
    result = engine.migrate(tenant, source, target, firewall)
    assert (result.success, result.reason) == (False, "target_too_small")
    assert (source.tenant_id, target.is_free, tenant.region_id) == ("a", True, 0)
    assert [(rule.tenant_id, rule.base_addr) for rule in firewall.rules] == [("a", 0x4000_0000)]
    assert engine.history == [result]


def test_migration_to_the_same_region_changes_nothing() -> None:
    """Migrating a tenant to the region it holds is refused."""
    engine, firewall = MigrationEngine(), BitstreamFirewall()
    region = _region(0, 0x4000_0000)
    region.state, region.tenant_id = RegionState.ALLOCATED, "a"
    tenant = Tenant(tenant_id="a", name="a", region_id=0, active=True)
    result = engine.migrate(tenant, region, region, firewall)
    assert (result.success, result.reason) == (False, "same_region")
    assert (region.state, region.tenant_id) == (RegionState.ALLOCATED, "a")


def test_the_rate_counts_spikes_after_the_first_sample_over_their_span() -> None:
    """Two samples of 10 spikes ten cycles apart give one spike per cycle."""
    meter = BandwidthMeter()
    meter.record("t", 10, 0)
    meter.record("t", 10, 10)
    assert meter.throughput("t") == 1.0


def test_the_rate_reads_only_the_window() -> None:
    """The base is the latest sample at or before the window's start; earlier samples are not counted."""
    meter = BandwidthMeter(window_cycles=100)
    for cycle in (0, 50, 100, 150):
        meter.record("t", 10, cycle)
    assert meter.throughput("t") == pytest.approx(20 / 100)


def test_the_rate_of_one_sample_is_zero() -> None:
    """With no interval to divide by, the rate is 0.0."""
    meter = BandwidthMeter()
    meter.record("t", 7, 1000)
    assert meter.throughput("t") == 0.0


@pytest.mark.parametrize("cycle", [100, 50])
def test_a_sample_not_later_than_the_tenant_s_last_is_refused(cycle: int) -> None:
    """Cycles increase per tenant; a repeated or earlier one is refused and the history is kept."""
    meter = BandwidthMeter()
    meter.record("t", 10, 0)
    meter.record("t", 10, 100)
    with pytest.raises(ValueError, match="not later than"):
        meter.record("t", 10, cycle)
    assert meter.throughput("t") == pytest.approx(10 / 100)
    meter.record("u", 10, cycle)
    assert meter.throughput("u") == 0.0


def test_megabits_per_second_use_the_stated_packet_width_and_clock() -> None:
    """One spike per cycle of 32 bits at 200 MHz is 6400 Mbps."""
    meter = BandwidthMeter()
    meter.record("t", 10, 0)
    meter.record("t", 10, 10)
    assert meter.throughput_mbps("t", bits_per_spike=32, clock_hz=200e6) == pytest.approx(6400.0)
    assert meter.exceeds_quota("t", 6399.0, bits_per_spike=32, clock_hz=200e6) is True
    assert meter.exceeds_quota("t", 6400.0, bits_per_spike=32, clock_hz=200e6) is False


@pytest.mark.parametrize(
    ("bits", "clock", "limit"),
    [
        (0, 200e6, 1.0),
        (True, 200e6, 1.0),
        (32.0, 200e6, 1.0),
        (32, 0.0, 1.0),
        (32, math.inf, 1.0),
        (32, True, 1.0),
        (32, 200e6, -1.0),
        (32, 200e6, math.nan),
        (32, 200e6, True),
    ],
)
def test_the_quota_refuses_units_it_cannot_use(bits: int, clock: float, limit: float) -> None:
    """A packet width that is not a positive int, a clock or a limit that is a bool, not finite, or out of range."""
    meter = BandwidthMeter()
    with pytest.raises(ValueError):
        meter.exceeds_quota("t", limit, bits_per_spike=bits, clock_hz=clock)


def test_every_tenant_of_a_priority_plan_receives_cycles() -> None:
    """Three realtime tenants share one half; the plan stays within the total and no tenant gets nothing."""
    tenants = [
        Tenant(tenant_id=f"rt{i}", name=f"rt{i}", priority=TenantPriority.REALTIME)
        for i in range(3)
    ] + [Tenant(tenant_id=f"n{i}", name=f"n{i}") for i in range(2)]
    for index, tenant in enumerate(tenants):
        tenant.active, tenant.region_id = True, index
    slots = Scheduler(SchedulingPolicy.PRIORITY).generate_schedule(tenants, 1000)
    cycles = {slot.tenant_id: slot.duration_cycles for slot in slots}
    assert set(cycles) == {tenant.tenant_id for tenant in tenants}
    assert all(duration > 0 for duration in cycles.values())
    assert sum(cycles.values()) <= 1000
    assert [cycles[f"rt{i}"] for i in range(3)] == [1000 // 2 // 3] * 3

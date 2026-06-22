# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic Hypervisor Tests

from typing import List

import numpy as np
import pytest

from sc_neurocore.hypervisor.hypervisor import (
    AuditEntry,
    AuditEventType,
    BandwidthMeter,
    BitstreamFirewall,
    FirewallRule,
    HWRegion,
    Hypervisor,
    HypervisorConfig,
    MigrationEngine,
    MigrationThrottle,
    PreemptionManager,
    RegionHealth,
    RegionState,
    ResourceAccounting,
    SLAMonitor,
    Scheduler,
    SchedulingPolicy,
    SecurityAuditLog,
    Tenant,
    TenantPriority,
    TenantState,
    admission_check,
    select_region_multi_die,
    verify_isolation,
)


# ── helpers ──────────────────────────────────────────────────────────


def _region(rid: int = 0, neurons: int = 1024, base: int = 0x4000_0000) -> HWRegion:
    return HWRegion(
        region_id=rid,
        num_neurons=neurons,
        num_synapses=neurons * 16,
        axi_base_addr=base,
        axi_size=0x1000,
        die_id=0,
    )


def _tenant(
    tid: str = "t0", name: str = "test", prio: TenantPriority = TenantPriority.NORMAL
) -> Tenant:
    return Tenant(tenant_id=tid, name=name, priority=prio)


# ── HWRegion Tests ───────────────────────────────────────────────────


class TestHWRegion:
    def test_defaults(self):
        r = _region()
        assert r.is_free
        assert r.state == RegionState.FREE

    def test_axi_end_addr(self):
        r = _region(base=0x1000)
        assert r.axi_end_addr == 0x1000 + 0x1000

    def test_contains_addr(self):
        r = _region(base=0x1000)
        assert r.contains_addr(0x1000) is True
        assert r.contains_addr(0x1FFF) is True
        assert r.contains_addr(0x2000) is False
        assert r.contains_addr(0x0FFF) is False


# ── TenantState Tests ────────────────────────────────────────────────


class TestTenantState:
    def test_checksum(self):
        ts = TenantState(
            neuron_voltages=np.array([1.0, 2.0]),
            synapse_weights=np.array([0.5, 0.3]),
            lfsr_state=42,
            timestep=100,
        )
        cs = ts.compute_checksum()
        assert len(cs) == 16

    def test_checksum_deterministic(self):
        ts = TenantState(lfsr_state=42, timestep=100)
        assert ts.compute_checksum() == ts.compute_checksum()

    def test_checksum_differs(self):
        ts1 = TenantState(lfsr_state=42, timestep=100)
        ts2 = TenantState(lfsr_state=43, timestep=100)
        assert ts1.compute_checksum() != ts2.compute_checksum()


# ── BitstreamFirewall Tests ──────────────────────────────────────────


class TestBitstreamFirewall:
    def test_allow_read(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x1050) is True

    def test_deny_out_of_range(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t0", 0x2000) is False

    def test_deny_cross_tenant(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        assert fw.check_access("t1", 0x1050) is False

    def test_deny_write(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, write_allowed=False))
        assert fw.check_access("t0", 0x1050, is_write=True) is False

    def test_deny_read(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100, read_allowed=False))
        assert fw.check_access("t0", 0x1050) is False

    def test_violation_logged(self):
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        assert fw.violation_count == 1

    def test_remove_rules(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        removed = fw.remove_tenant_rules("t0")
        assert removed == 1
        assert fw.check_access("t0", 0x1050) is False

    def test_clear_violations(self):
        fw = BitstreamFirewall()
        fw.check_access("t0", 0x1000)
        fw.clear_violations()
        assert fw.violation_count == 0


# ── Scheduler Tests ──────────────────────────────────────────────────


class TestScheduler:
    def _tenants(self) -> List[Tenant]:
        t0 = _tenant("t0", prio=TenantPriority.NORMAL)
        t0.active = True
        t0.region_id = 0
        t1 = _tenant("t1", prio=TenantPriority.HIGH)
        t1.active = True
        t1.region_id = 1
        return [t0, t1]

    def test_round_robin(self):
        sched = Scheduler(SchedulingPolicy.ROUND_ROBIN)
        sched.time_quantum_cycles = 100
        slots = sched.generate_schedule(self._tenants(), 400)
        assert len(slots) == 4
        assert slots[0].tenant_id != slots[1].tenant_id

    def test_priority(self):
        sched = Scheduler(SchedulingPolicy.PRIORITY)
        slots = sched.generate_schedule(self._tenants(), 1000)
        assert len(slots) > 0
        # Higher priority tenant should get scheduled first
        assert slots[0].tenant_id == "t1"

    def test_fair_share(self):
        tenants = self._tenants()
        tenants[0].qos.min_compute_share = 0.3
        tenants[1].qos.min_compute_share = 0.7
        sched = Scheduler(SchedulingPolicy.FAIR_SHARE)
        slots = sched.generate_schedule(tenants, 1000)
        total_t1 = sum(s.duration_cycles for s in slots if s.tenant_id == "t1")
        assert total_t1 > 500

    def test_edf(self):
        tenants = self._tenants()
        tenants[0].qos.max_latency_us = 1000
        tenants[1].qos.max_latency_us = 100
        sched = Scheduler(SchedulingPolicy.EDF)
        slots = sched.generate_schedule(tenants, 1000)
        assert slots[0].tenant_id == "t1"  # Lower latency first

    def test_empty_tenants(self):
        sched = Scheduler()
        assert sched.generate_schedule([], 1000) == []

    def test_priority_realtime_gets_half(self):
        # A realtime tenant is granted a fixed 50% slice under priority
        # scheduling rather than the even per-tenant share.
        rt = _tenant("rt", prio=TenantPriority.REALTIME)
        rt.active = True
        rt.region_id = 0
        normal = _tenant("nm", prio=TenantPriority.NORMAL)
        normal.active = True
        normal.region_id = 1
        sched = Scheduler(SchedulingPolicy.PRIORITY)
        slots = sched.generate_schedule([rt, normal], 1000)
        rt_slot = next(s for s in slots if s.tenant_id == "rt")
        assert rt_slot.duration_cycles == 500

    def test_slot_continuity(self):
        sched = Scheduler(SchedulingPolicy.ROUND_ROBIN)
        sched.time_quantum_cycles = 100
        slots = sched.generate_schedule(self._tenants(), 400)
        for i in range(1, len(slots)):
            assert slots[i].start_cycle == slots[i - 1].end_cycle


# ── MigrationEngine Tests ────────────────────────────────────────────


class TestMigrationEngine:
    def test_checkpoint(self):
        me = MigrationEngine()
        t = _tenant()
        t.state = TenantState(
            neuron_voltages=np.array([1.0]),
            lfsr_state=42,
            timestep=10,
        )
        state = me.checkpoint(t)
        assert state.checksum != ""

    def test_restore(self):
        me = MigrationEngine()
        t = _tenant()
        state = TenantState(lfsr_state=42, timestep=10)
        state.compute_checksum()
        assert me.restore(t, state) is True
        assert t.state is not None

    def test_checkpoint_initialises_missing_state(self):
        # A tenant that has never run has no state; checkpointing one must
        # materialise a fresh TenantState rather than dereference None.
        me = MigrationEngine()
        t = _tenant()
        assert t.state is None
        state = me.checkpoint(t)
        assert isinstance(state, TenantState)
        assert t.state is state

    def test_restore_rejects_tampered_state(self):
        # If the stored checksum no longer matches the recomputed one (the state
        # was altered after checkpointing), restore must refuse it.
        me = MigrationEngine()
        t = _tenant()
        state = TenantState(lfsr_state=42, timestep=10)
        state.compute_checksum()
        state.timestep = 99  # tamper after the checksum was sealed
        assert me.restore(t, state) is False

    def test_migrate_success(self):
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState(lfsr_state=42, timestep=10)
        t.region_id = 0
        src = _region(rid=0)
        src.state = RegionState.ALLOCATED
        src.tenant_id = "t0"
        dst = _region(rid=1, base=0x5000_0000)
        result = me.migrate(t, src, dst, fw)
        assert result.success is True
        assert t.region_id == 1
        assert src.is_free
        assert dst.tenant_id == "t0"

    def test_migrate_target_busy(self):
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState()
        t.region_id = 0
        src = _region(rid=0)
        dst = _region(rid=1)
        dst.state = RegionState.ALLOCATED
        result = me.migrate(t, src, dst, fw)
        assert result.success is False
        assert result.reason == "target_not_free"

    def test_migration_history(self):
        me = MigrationEngine()
        fw = BitstreamFirewall()
        t = _tenant()
        t.state = TenantState()
        t.region_id = 0
        src = _region(rid=0)
        dst = _region(rid=1, base=0x5000_0000)
        me.migrate(t, src, dst, fw)
        assert len(me.history) == 1


# ── Hypervisor Tests ─────────────────────────────────────────────────


class TestHypervisor:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=2048, base=0x5000_0000))
        hv.add_region(_region(2, neurons=512, base=0x6000_0000))
        return hv

    def test_register_tenant(self):
        hv = self._setup()
        t = _tenant()
        assert hv.register_tenant(t) is True
        assert "t0" in hv.tenants

    def test_register_duplicate(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        assert hv.register_tenant(_tenant("t0")) is False

    def test_allocate(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        rid = hv.allocate("t0")
        assert rid is not None
        assert hv.tenants["t0"].active is True

    def test_allocate_respects_capacity(self):
        hv = self._setup()
        t = _tenant("t0")
        t.qos.max_neurons = 2000
        hv.register_tenant(t)
        rid = hv.allocate("t0")
        # Only region 1 has 2048 neurons
        assert rid == 1

    def test_deallocate(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        assert hv.deallocate("t0") is True
        assert hv.tenants["t0"].active is False

    def test_remove_tenant(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        assert hv.remove_tenant("t0") is True
        assert "t0" not in hv.tenants

    def test_schedule(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        hv.allocate("t0")
        hv.allocate("t1")
        slots = hv.schedule(10000)
        assert len(slots) > 0

    def test_firewall_access(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        region = hv.regions[rid]
        assert hv.check_access("t0", region.axi_base_addr) is True
        assert hv.check_access("t0", 0xDEAD_BEEF) is False

    def test_firewall_cross_tenant(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        hv.allocate("t0")
        hv.allocate("t1")
        r0 = hv.regions[hv.tenants["t0"].region_id]
        assert hv.check_access("t1", r0.axi_base_addr) is False

    def test_migrate(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        old_rid = hv.tenants["t0"].region_id
        free_rid = [r for r in hv.regions if hv.regions[r].is_free][0]
        hv.tenants["t0"].state = TenantState(lfsr_state=42)
        result = hv.migrate("t0", free_rid)
        assert result.success is True
        assert hv.tenants["t0"].region_id == free_rid

    def test_status(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        st = hv.status()
        assert st["total_regions"] == 3
        assert st["active_tenants"] == 1
        assert st["free_regions"] == 2

    def test_tenant_report(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0", name="BCI"))
        hv.allocate("t0")
        rpt = hv.tenant_report("t0")
        assert rpt is not None
        assert rpt["name"] == "BCI"
        assert rpt["active"] is True

    def test_max_tenants(self):
        hv = Hypervisor(HypervisorConfig(max_tenants=2))
        hv.add_region(_region(0))
        hv.add_region(_region(1))
        hv.register_tenant(_tenant("t0"))
        hv.register_tenant(_tenant("t1"))
        assert hv.register_tenant(_tenant("t2")) is False

    def test_allocate_unknown_tenant(self):
        hv = self._setup()
        assert hv.allocate("ghost") is None

    def test_allocate_no_region_fits(self):
        hv = self._setup()
        t = _tenant("big")
        t.qos.max_neurons = 100_000  # larger than every region
        hv.register_tenant(t)
        assert hv.allocate("big") is None

    def test_migrate_unknown_tenant_not_found(self):
        hv = self._setup()
        result = hv.migrate("ghost", 1)
        assert result.success is False
        assert result.reason == "not_found"

    def test_migrate_invalid_target_region(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        result = hv.migrate("t0", 999)  # no such region
        assert result.success is False
        assert result.reason == "invalid_region"

    def test_tenant_report_unknown_returns_none(self):
        hv = self._setup()
        assert hv.tenant_report("ghost") is None


# ── Utilisation Tests ────────────────────────────────────────────────


class TestUtilisation:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=2048, base=0x5000_0000))
        return hv

    def test_utilisation_empty(self):
        hv = self._setup()
        util = hv.compute_utilisation()
        assert util[0] == 0.0
        assert util[1] == 0.0

    def test_utilisation_allocated(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        util = hv.compute_utilisation()
        rid = hv.tenants["t0"].region_id
        assert util[rid] > 0.0

    def test_utilisation_bounded(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        util = hv.compute_utilisation()
        for v in util.values():
            assert 0.0 <= v <= 1.0

    def test_utilisation_unassigned_non_free_region_is_idle(self):
        # A region that is not free yet carries no tenant_id (e.g. faulted out of
        # service) contributes zero utilisation.
        hv = self._setup()
        hv.regions[0].state = RegionState.FAULTED
        util = hv.compute_utilisation()
        assert util[0] == 0.0

    def test_utilisation_orphaned_region_is_full(self):
        # A region still tagged with a tenant_id whose tenant record is gone
        # (e.g. removed without deallocation) is treated as fully utilised.
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        del hv.tenants["t0"]  # region keeps its tenant_id, tenant lookup misses
        util = hv.compute_utilisation()
        assert util[rid] == 1.0


# ── Overcommit Tests ─────────────────────────────────────────────────


class TestOvercommit:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=512))
        return hv

    def test_no_overcommit(self):
        hv = self._setup()
        t = _tenant("t0")
        t.qos.max_neurons = 256
        hv.register_tenant(t)
        hv.allocate("t0")
        assert hv.check_overcommit() is False

    def test_overcommit_detected(self):
        hv = self._setup()
        for i in range(3):
            t = _tenant(f"t{i}")
            t.qos.max_neurons = 512
            hv.register_tenant(t)
            t.active = True
        assert hv.check_overcommit() is True


# ── Region Fault Tests ───────────────────────────────────────────────


class TestRegionFault:
    def _setup(self) -> Hypervisor:
        hv = Hypervisor()
        hv.add_region(_region(0, neurons=1024, base=0x4000_0000))
        hv.add_region(_region(1, neurons=1024, base=0x5000_0000))
        return hv

    def test_no_faults_initially(self):
        hv = self._setup()
        assert hv.get_faulted_regions() == []

    def test_mark_faulted(self):
        hv = self._setup()
        hv.mark_region_faulted(0)
        assert 0 in hv.get_faulted_regions()
        assert hv.regions[0].state == RegionState.FAULTED

    def test_fault_evicts_tenant(self):
        hv = self._setup()
        hv.register_tenant(_tenant("t0"))
        hv.allocate("t0")
        rid = hv.tenants["t0"].region_id
        hv.mark_region_faulted(rid)
        assert hv.tenants["t0"].active is False
        assert hv.tenants["t0"].region_id is None

    def test_fault_nonexistent_region(self):
        hv = self._setup()
        assert hv.mark_region_faulted(999) is False


# ── No-Firewall Mode Test ────────────────────────────────────────────


class TestNoFirewall:
    def test_firewall_disabled(self):
        hv = Hypervisor(HypervisorConfig(enable_firewall=False))
        hv.add_region(_region(0))
        assert hv.check_access("anyone", 0xDEAD_BEEF) is True


# ── Bandwidth Meter Tests (Gap 1) ─────────────────────────────────────


class TestBandwidthMeter:
    def test_record_and_throughput(self):
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        bm.record("t0", 200, 2000)
        assert bm.throughput("t0") > 0

    def test_zero_throughput(self):
        bm = BandwidthMeter()
        assert bm.throughput("t_none") == 0.0

    def test_throughput_single_record_returns_raw_count(self):
        # With only one sample there is no time span to divide by, so the
        # throughput is reported as the raw spike count.
        bm = BandwidthMeter()
        bm.record("t0", 100, 1000)
        assert bm.throughput("t0") == 100.0

    def test_exceeds_quota(self):
        bm = BandwidthMeter()
        bm.record("t0", 1000, 100)
        bm.record("t0", 1000, 200)
        assert bm.exceeds_quota("t0", 0.001) is True


# ── Preemption Manager Tests (Gap 2) ─────────────────────────────────


class TestPreemptionManager:
    def test_preempt(self):
        pm = PreemptionManager()
        victim = _tenant("v")
        victim.active = True
        victim.region_id = 0
        victim.state = TenantState(lfsr_state=42)
        preemptor = _tenant("p")
        region = _region(0)
        region.tenant_id = "v"
        evt = pm.preempt(victim, preemptor, region, cycle=1000)
        assert evt.state_saved is True
        assert victim.active is False
        assert preemptor.region_id == 0

    def test_restore_preempted(self):
        pm = PreemptionManager()
        victim = _tenant("v")
        victim.state = TenantState(lfsr_state=99)
        preemptor = _tenant("p")
        region = _region(0)
        pm.preempt(victim, preemptor, region, cycle=0)
        assert pm.restore_preempted(victim) is True
        assert victim.state.lfsr_state == 99

    def test_restore_missing(self):
        pm = PreemptionManager()
        t = _tenant("x")
        assert pm.restore_preempted(t) is False


# ── SLA Monitor Tests (Gap 3) ─────────────────────────────────────────


class TestSLAMonitor:
    def test_latency_ok(self):
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_latency_us = 1000.0
        assert mon.check_latency(t, 500.0, 100) is None

    def test_latency_violation(self):
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_latency_us = 1000.0
        v = mon.check_latency(t, 1500.0, 100)
        assert v is not None
        assert v.metric == "latency"
        assert mon.total_violations == 1

    def test_bandwidth_ok(self):
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_bandwidth_mbps = 100.0
        assert mon.check_bandwidth(t, 50.0, 10) is None

    def test_bandwidth_violation(self):
        mon = SLAMonitor()
        t = _tenant("t0")
        t.qos.max_bandwidth_mbps = 100.0
        v = mon.check_bandwidth(t, 200.0, 50)
        assert v is not None

    def test_violations_for(self):
        mon = SLAMonitor()
        t0 = _tenant("t0")
        t0.qos.max_latency_us = 10.0
        t1 = _tenant("t1")
        t1.qos.max_latency_us = 10.0
        mon.check_latency(t0, 100.0, 1)
        mon.check_latency(t1, 100.0, 2)
        assert len(mon.violations_for("t0")) == 1


# ── Multi-Die Region Selector Tests (Gap 4) ───────────────────────────


class TestMultiDieSelector:
    def test_prefers_die(self):
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
            1: HWRegion(1, 1024, 0, 0x2000, 0x1000, die_id=1),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=1)
        assert rid == 1

    def test_fallback_any_die(self):
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=5)
        assert rid == 0

    def test_no_fit(self):
        regions = {
            0: HWRegion(0, 64, 0, 0x1000, 0x1000, die_id=0),
        }
        assert select_region_multi_die(regions, 512) is None


# ── Resource Accounting Tests (Gap 5) ─────────────────────────────────


class TestResourceAccounting:
    def test_record_and_query(self):
        ra = ResourceAccounting()
        ra.record("t0", 10000, 500)
        ra.record("t0", 5000, 300)
        assert ra.total_cycles("t0") == 15000
        assert ra.total_spikes("t0") == 800

    def test_invoice(self):
        ra = ResourceAccounting()
        ra.record("t0", 1_000_000, 0)
        assert ra.invoice("t0", cost_per_cycle=1e-6) == pytest.approx(1.0)

    def test_unknown_tenant(self):
        ra = ResourceAccounting()
        assert ra.total_cycles("nobody") == 0


# ── Admission Control Tests (Gap 6) ───────────────────────────────────


class TestAdmissionControl:
    def test_admit_ok(self):
        t = _tenant("new")
        t.qos.max_neurons = 512
        regions = {0: _region(0, neurons=1024)}
        ok, msg = admission_check(t, regions, {})
        assert ok is True

    def test_reject_insufficient(self):
        t = _tenant("new")
        t.qos.max_neurons = 2048
        regions = {0: _region(0, neurons=512)}
        ok, msg = admission_check(t, regions, {})
        assert ok is False
        assert "insufficient" in msg

    def test_reject_no_single_region_large_enough(self):
        # Aggregate free capacity is sufficient, but no single region can hold
        # the tenant: admission is refused because a tenant cannot be split.
        t = _tenant("new")
        t.qos.max_neurons = 800
        regions = {
            0: _region(0, neurons=512, base=0x4000_0000),
            1: _region(1, neurons=512, base=0x5000_0000),
        }
        ok, msg = admission_check(t, regions, {})
        assert ok is False
        assert msg == "no_single_region_large_enough"


# ── Region Health Tests (Gap 7) ───────────────────────────────────────


class TestRegionHealth:
    def test_healthy_default(self):
        rh = RegionHealth(region_id=0)
        assert rh.health_score == pytest.approx(1.0)
        assert not rh.is_degraded

    def test_degraded_by_errors(self):
        rh = RegionHealth(region_id=0, error_count=5)
        assert rh.health_score < 0.8
        assert rh.is_degraded

    def test_temperature_penalty(self):
        rh = RegionHealth(region_id=0, temperature_c=100.0)
        assert rh.health_score < 1.0

    def test_record_error(self):
        rh = RegionHealth(region_id=0)
        rh.record_error()
        assert rh.error_count == 1


# ── Security Audit Log Tests (Gap 8) ──────────────────────────────────


class TestSecurityAuditLog:
    def test_log_and_count(self):
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.TENANT_REGISTERED, "t0", "registered"))
        assert log.count == 1

    def test_query_by_type(self):
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "bad access"))
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "migrated"))
        denied = log.query(event_type=AuditEventType.ACCESS_DENIED)
        assert len(denied) == 1

    def test_query_by_tenant(self):
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t0", "a"))
        log.log(AuditEntry(AuditEventType.ACCESS_DENIED, "t1", "b"))
        assert len(log.query(tenant_id="t0")) == 1

    def test_checksum(self):
        log = SecurityAuditLog()
        log.log(AuditEntry(AuditEventType.MIGRATION, "t0", "x"))
        cs = log.checksum()
        assert len(cs) == 16


# ── Isolation Verification Tests (Gap 9) ──────────────────────────────


class TestIsolationVerification:
    def test_no_overlap(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x100))
        fw.add_rule(FirewallRule("t1", 0x2000, 0x100))
        violations = verify_isolation(fw, {})
        assert violations == []

    def test_overlap_detected(self):
        fw = BitstreamFirewall()
        fw.add_rule(FirewallRule("t0", 0x1000, 0x200))
        fw.add_rule(FirewallRule("t1", 0x1100, 0x200))
        violations = verify_isolation(fw, {})
        assert len(violations) == 1
        assert "overlap" in violations[0]


# ── Migration Throttle Tests (Gap 10) ─────────────────────────────────


class TestMigrationThrottle:
    def test_initial_allow(self):
        mt = MigrationThrottle(max_per_window=3)
        assert mt.allow() is True

    def test_throttled(self):
        mt = MigrationThrottle(max_per_window=2, window_ns=10_000_000_000)
        mt.record()
        mt.record()
        assert mt.allow() is False

    def test_recent_count(self):
        mt = MigrationThrottle()
        mt.record()
        mt.record()
        assert mt.recent_count == 2

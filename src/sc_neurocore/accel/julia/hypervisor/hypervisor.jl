# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for hypervisor/hypervisor

module HypervisorAccel

using Statistics, LinearAlgebra

mutable struct MigrationThrottleState
    region_id::Float64
    num_neurons::Float64
    num_synapses::Float64
    axi_base_addr::Float64
    axi_size::Float64
    die_id::Float64
    state::Float64
    tenant_id::Float64
    utilisation::Float64
    max_bandwidth_mbps::Float64
    max_latency_us::Float64
    min_compute_share::Float64
    max_neurons::Float64
    max_synapses::Float64
    preemptible::Float64
end

function MigrationThrottleState()
    MigrationThrottleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1000.0, 0.1, 1024.0, 16384.0, 1.0)
end

function axi_end_addr(s::MigrationThrottleState)
    return s.axi_base_addr + s.axi_size
end

function is_free(s::MigrationThrottleState)
    return s.state == RegionState.FREE
end

function contains_addr(s::MigrationThrottleState, addr)
    return s.axi_base_addr <= addr < s.axi_end_addr
end

function compute_checksum(s::MigrationThrottleState)
    h = hashlib.sha256()
    if s.neuron_voltages is ! nothing
        h.update(s.neuron_voltages.tobytes())
    if s.synapse_weights is ! nothing
        h.update(s.synapse_weights.tobytes())
    h.update(s.lfsr_state.to_bytes(4, "little"))
    h.update(s.timestep.to_bytes(4, "little"))
    s.checksum = h.hexdigest()[:16]
    return s.checksum
end

function end_addr(s::MigrationThrottleState)
    return s.base_addr + s.size
end

function add_rule(s::MigrationThrottleState, rule)
    s.rules = push!(, rule)
end

function remove_tenant_rules(s::MigrationThrottleState, tenant_id)
    before = length(s.rules)
    s.rules = [r for r in s.rules if r.tenant_id != tenant_id]
    return before - length(s.rules)
end

function check_access(s::MigrationThrottleState, tenant_id, addr, is_write)
    for rule in s.rules
        if rule.tenant_id != tenant_id
            continue
        if rule.base_addr <= addr < rule.end_addr
            if is_write && ! rule.write_allowed
                s._log_violation(tenant_id, addr, "write_denied")
                return false
            if ! is_write && ! rule.read_allowed
                s._log_violation(tenant_id, addr, "read_denied")
                return false
            return true
    s._log_violation(tenant_id, addr, "no_rule")
    return false
end

function _log_violation(s::MigrationThrottleState, tenant_id, addr, reason)
    if length(s.violations) < s.max_violations
        s.violations = push!(, 
            {
                "tenant_id": tenant_id,
                "addr": hex(addr),
                "reason": reason,
                "timestamp_ns": time.time_ns(),
            }
        )
end

function violation_count(s::MigrationThrottleState)
    return length(s.violations)
end

function clear_violations(s::MigrationThrottleState)
    s.violations.clear()
end

function end_cycle(s::MigrationThrottleState)
    return s.start_cycle + s.duration_cycles
end

function generate_schedule(s::MigrationThrottleState, tenants, num_cycles)
    if ! tenants
        return []
    active = [t for t in tenants if t.active && t.region_id is ! nothing]
    if ! active
        return []
    if s.policy == SchedulingPolicy.ROUND_ROBIN
        return s._round_robin(active, num_cycles)
    elseif s.policy == SchedulingPolicy.PRIORITY
        return s._priority(active, num_cycles)
    elseif s.policy == SchedulingPolicy.FAIR_SHARE
        return s._fair_share(active, num_cycles)
    elseif s.policy == SchedulingPolicy.EDF
        return s._edf(active, num_cycles)
    return []
end

function _round_robin(s::MigrationThrottleState, tenants, total)
    slots = []
    cycle = 0
    idx = 0
    while cycle < total
        t = tenants[idx % length(tenants)]
        dur = min(s.time_quantum_cycles, total - cycle)
        slots = push!(, ScheduleSlot(t.tenant_id, t.region_id || 0, cycle, dur))
        cycle += dur
        idx += 1
    s.schedule = slots
    return slots
end

function _priority(s::MigrationThrottleState, tenants, total)
    sorted_t = sorted(tenants, key=lambda t: t.priority.value)
    slots = []
    cycle = 0
    for t in sorted_t
        share = max(1, total // length(tenants))
        if t.priority == TenantPriority.REALTIME
            share = total // 2  # Realtime gets 50%
        dur = min(share, total - cycle)
        if dur > 0
            slots = push!(, ScheduleSlot(t.tenant_id, t.region_id || 0, cycle, dur))
            cycle += dur
    s.schedule = slots
    return slots
end

function _fair_share(s::MigrationThrottleState, tenants, total)
    total_share = sum(t.qos.min_compute_share for t in tenants)
    slots = []
    cycle = 0
    for t in tenants
        frac = t.qos.min_compute_share / total_share if total_share > 0 else 1.0 / length(tenants)
        dur = int(total * frac)
        dur = min(dur, total - cycle)
        if dur > 0
            slots = push!(, ScheduleSlot(t.tenant_id, t.region_id || 0, cycle, dur))
            cycle += dur
    s.schedule = slots
    return slots
end

function _edf(s::MigrationThrottleState, tenants, total)
    sorted_t = sorted(tenants, key=lambda t: t.qos.max_latency_us)
    return s._round_robin(sorted_t, total)
end

function checkpoint(s::MigrationThrottleState, tenant)
    if tenant.state is nothing
        tenant.state = TenantState()
    tenant.state.compute_checksum()
    return tenant.state
end

function restore(s::MigrationThrottleState, tenant, state)
    verify = state.checksum
    recomputed = state.compute_checksum()
    if verify && verify != recomputed
        return false
    tenant.state = state
    return true
end

function migrate(s::MigrationThrottleState)
    self,
    tenant: Tenant,
    source: HWRegion,
    target: HWRegion,
    firewall: BitstreamFirewall,
    ) -> MigrationResult
    start = time.time_ns()
    # 1. Checkpoint
    state = s.checkpoint(tenant)
    checksum = state.checksum
    # 2. Free source
    source.state = RegionState.FREE
    source.tenant_id = nothing
    # 3. Allocate target
    if ! target.is_free
        result = MigrationResult(
            false,
            tenant.tenant_id,
            source.region_id,
            target.region_id,
            reason="target_not_free",
        )
        s.history = push!(, result)
        return result
    target.state = RegionState.ALLOCATED
    target.tenant_id = tenant.tenant_id
    # 4. Update firewall
    firewall.remove_tenant_rules(tenant.tenant_id)
    firewall.add_rule(
        FirewallRule(
            tenant.tenant_id,
            target.axi_base_addr,
            target.axi_size,
        )
    )
    # 5. Restore state
    success = s.restore(tenant, state)
    tenant.region_id = target.region_id
    elapsed = time.time_ns() - start
    result = MigrationResult(
        success,
        tenant.tenant_id,
        source.region_id,
        target.region_id,
        checksum,
        elapsed,
    )
    s.history = push!(, result)
    return result
end

function add_region(s::MigrationThrottleState, region)
    s.regions[region.region_id] = region
end

function register_tenant(s::MigrationThrottleState, tenant)
    if length(s.tenants) >= s.config.max_tenants
        return false
    if tenant.tenant_id in s.tenants
        return false
    tenant.created_ns = time.time_ns()
    s.tenants[tenant.tenant_id] = tenant
    return true
end

function allocate(s::MigrationThrottleState, tenant_id)
    tenant = s.tenants.get(tenant_id)
    if tenant is nothing
        return nothing
    # Find a free region that fits the QoS
    for rid, region in s.regions.items()
        if ! region.is_free
            continue
        if region.num_neurons < tenant.qos.max_neurons
            continue
        # Allocate
        region.state = RegionState.ALLOCATED
        region.tenant_id = tenant_id
        tenant.region_id = rid
        tenant.active = true
        # Set up firewall
        if s.config.enable_firewall
            s.firewall.add_rule(
                FirewallRule(
                    tenant_id,
                    region.axi_base_addr,
                    region.axi_size,
                )
            )
        return rid
    return nothing
end

function deallocate(s::MigrationThrottleState, tenant_id)
    tenant = s.tenants.get(tenant_id)
    if tenant is nothing || tenant.region_id is nothing
        return false
    region = s.regions.get(tenant.region_id)
    if region is ! nothing
        region.state = RegionState.FREE
        region.tenant_id = nothing
    s.firewall.remove_tenant_rules(tenant_id)
    tenant.region_id = nothing
    tenant.active = false
    return true
end

function remove_tenant(s::MigrationThrottleState, tenant_id)
    s.deallocate(tenant_id)
    return s.tenants.pop(tenant_id, nothing) is ! nothing
end

function schedule(s::MigrationThrottleState, num_cycles)
    active = [t for t in s.tenants.values() if t.active]
    return s.scheduler.generate_schedule(active, num_cycles)
end

function migrate(s::MigrationThrottleState, tenant_id, target_region_id)
    tenant = s.tenants.get(tenant_id)
    if tenant is nothing || tenant.region_id is nothing
        return MigrationResult(false, tenant_id || "", -1, target_region_id, reason="not_found")
    source = s.regions.get(tenant.region_id)
    target = s.regions.get(target_region_id)
    if source is nothing || target is nothing
        return MigrationResult(false, tenant_id, -1, target_region_id, reason="invalid_region")
    return s.migration_engine.migrate(tenant, source, target, s.firewall)
end

function check_access(s::MigrationThrottleState, tenant_id, addr, is_write)
    if ! s.config.enable_firewall
        return true
    return s.firewall.check_access(tenant_id, addr, is_write)
end

function status(s::MigrationThrottleState)
    free_regions = sum(1 for r in s.regions.values() if r.is_free)
    active_tenants = sum(1 for t in s.tenants.values() if t.active)
    return {
        "total_regions": length(s.regions),
        "free_regions": free_regions,
        "total_tenants": length(s.tenants),
        "active_tenants": active_tenants,
        "firewall_violations": s.firewall.violation_count,
        "migrations": length(s.migration_engine.history),
        "scheduling_policy": s.config.scheduling_policy.value,
    }
end

function tenant_report(s::MigrationThrottleState, tenant_id)
    t = s.tenants.get(tenant_id)
    if t is nothing
        return nothing
    return {
        "tenant_id": t.tenant_id,
        "name": t.name,
        "priority": t.priority.value,
        "region_id": t.region_id,
        "active": t.active,
        "total_spikes": t.total_spikes,
        "total_cycles": t.total_cycles,
        "qos_bandwidth_mbps": t.qos.max_bandwidth_mbps,
        "qos_latency_us": t.qos.max_latency_us,
    }
end

function compute_utilisation(s::MigrationThrottleState)
    result = {}
    for rid, region in s.regions.items()
        if region.is_free
            result[rid] = 0.0
        elseif region.tenant_id
            tenant = s.tenants.get(region.tenant_id)
            if tenant && tenant.qos
                result[rid] = min(1.0, tenant.qos.max_neurons / max(region.num_neurons, 1))
            else
                result[rid] = 1.0
        else
            result[rid] = 0.0
    return result
end

function check_overcommit(s::MigrationThrottleState)
    total_neurons_needed = sum(t.qos.max_neurons for t in s.tenants.values() if t.active)
    total_neurons_available = sum(r.num_neurons for r in s.regions.values())
    return total_neurons_needed > total_neurons_available
end

function get_faulted_regions(s::MigrationThrottleState)
    return [rid for rid, r in s.regions.items() if r.state == RegionState.FAULTED]
end

function mark_region_faulted(s::MigrationThrottleState, region_id)
    region = s.regions.get(region_id)
    if region is nothing
        return false
    if region.tenant_id
        s.deallocate(region.tenant_id)
    region.state = RegionState.FAULTED
    return true
end

function record(s::MigrationThrottleState, tenant_id, spike_count, cycle)
    if tenant_id ! in s._counters
        s._counters[tenant_id] = []
        s._timestamps[tenant_id] = []
    s._counters[tenant_id] = push!(, spike_count)
    s._timestamps[tenant_id] = push!(, cycle)
end

function throughput(s::MigrationThrottleState, tenant_id)
    if tenant_id ! in s._counters || ! s._counters[tenant_id]
        return 0.0
    entries = s._counters[tenant_id]
    total_spikes = sum(entries[-100:])
    if length(entries) < 2
        return float(total_spikes)
    ts = s._timestamps[tenant_id]
    span = max(1, ts[-1] - ts[max(0, length(ts) - 100)])
    return total_spikes / span
end

function exceeds_quota(s::MigrationThrottleState, tenant_id, max_mbps)
    return s.throughput(tenant_id) > max_mbps
end

function preempt(s::MigrationThrottleState)
    self,
    victim: Tenant,
    preemptor: Tenant,
    region: HWRegion,
    cycle: int,
    ) -> PreemptionEvent
    state_saved = false
    if victim.state is ! nothing
        victim.state.compute_checksum()
        s.saved_states[victim.tenant_id] = victim.state
        state_saved = true
    victim.active = false
    victim.region_id = nothing
    region.tenant_id = preemptor.tenant_id
    preemptor.region_id = region.region_id
    preemptor.active = true
    evt = PreemptionEvent(victim.tenant_id, preemptor.tenant_id, cycle, state_saved)
    s.events = push!(, evt)
    return evt
end

function restore_preempted(s::MigrationThrottleState, tenant)
    if tenant.tenant_id ! in s.saved_states
        return false
    tenant.state = s.saved_states.pop(tenant.tenant_id)
    return true
end

function check_latency(s::MigrationThrottleState)
    self, tenant: Tenant, measured_us: float, cycle: int
    ) -> Optional[SLAViolation]
    if measured_us > tenant.qos.max_latency_us
        v = SLAViolation(
            tenant.tenant_id, "latency", measured_us, tenant.qos.max_latency_us, cycle
        )
        s.violations = push!(, v)
        return v
    return nothing
end

function check_bandwidth(s::MigrationThrottleState)
    self, tenant: Tenant, measured_mbps: float, cycle: int
    ) -> Optional[SLAViolation]
    if measured_mbps > tenant.qos.max_bandwidth_mbps
        v = SLAViolation(
            tenant.tenant_id, "bandwidth", measured_mbps, tenant.qos.max_bandwidth_mbps, cycle
        )
        s.violations = push!(, v)
        return v
    return nothing
end

function total_violations(s::MigrationThrottleState)
    return length(s.violations)
end

function violations_for(s::MigrationThrottleState, tenant_id)
    return [v for v in s.violations if v.tenant_id == tenant_id]
end

function select_region_multi_die(regions, min_neurons, preferred_die)
    regions: Dict[int, HWRegion],
    min_neurons: int,
    preferred_die: Optional[int] = nothing,
    ) -> Optional[int]
    candidates = [
        (rid, r) for rid, r in regions.items() if r.is_free && r.num_neurons >= min_neurons
    ]
    if ! candidates
        return nothing
    if preferred_die is ! nothing
        on_die = [(rid, r) for rid, r in candidates if r.die_id == preferred_die]
        if on_die
            return min(on_die, key=lambda x: x[1].num_neurons)[0]
    return min(candidates, key=lambda x: x[1].num_neurons)[0]
end

function record(s::MigrationThrottleState, tenant_id, cycles, spikes)
    r = UsageRecord(tenant_id, cycles, spikes, time.time_ns())
    s.records = push!(, r)
    if tenant_id ! in s._totals
        s._totals[tenant_id] = {"cycles": 0, "spikes": 0}
    s._totals[tenant_id]["cycles"] += cycles
    s._totals[tenant_id]["spikes"] += spikes
end

function total_cycles(s::MigrationThrottleState, tenant_id)
    return s._totals.get(tenant_id, {}).get("cycles", 0)
end

function total_spikes(s::MigrationThrottleState, tenant_id)
    return s._totals.get(tenant_id, {}).get("spikes", 0)
end

function invoice(s::MigrationThrottleState, tenant_id, cost_per_cycle)
    return s.total_cycles(tenant_id) * cost_per_cycle
end

function admission_check(tenant, regions, existing_tenants)
    tenant: Tenant,
    regions: Dict[int, HWRegion],
    existing_tenants: Dict[str, Tenant],
    ) -> Tuple[bool, str]
    required = tenant.qos.max_neurons
    free_capacity = sum(r.num_neurons for r in regions.values() if r.is_free)
    if required > free_capacity
        return false, f"insufficient_neurons: need={required}, free={free_capacity}"
    if any(r.num_neurons >= required for r in regions.values() if r.is_free)
        return true, "admitted"
    return false, "no_single_region_large_enough"
end

function health_score(s::MigrationThrottleState)
    temp_pen = max(0, (s.temperature_c - 85)) * 0.01
    age_pen = s.age_hours / 100_000 * 0.1
    err_pen = min(s.error_count * 0.05, 0.5)
    return max(0.0, 1.0 - temp_pen - age_pen - err_pen)
end

function is_degraded(s::MigrationThrottleState)
    return s.health_score < 0.8
end

function record_error(s::MigrationThrottleState)
    s.error_count += 1
end

function log(s::MigrationThrottleState, event)
    s.entries = push!(, event)
end

function query(s::MigrationThrottleState)
    self, event_type: Optional[AuditEventType] = nothing, tenant_id: Optional[str] = nothing
    ) -> List[AuditEntry]
    results = list(s.entries)
    if event_type is ! nothing
        results = [e for e in results if e.event_type == event_type]
    if tenant_id is ! nothing
        results = [e for e in results if e.tenant_id == tenant_id]
    return results
end

function count(s::MigrationThrottleState)
    return length(s.entries)
end

function checksum(s::MigrationThrottleState)
    h = hashlib.sha256()
    for entry in s.entries
        h.update(f"{entry.event_type.value}:{entry.tenant_id}:{entry.timestamp_ns}".encode())
    return h.hexdigest()[:16]
end

function verify_isolation(firewall, regions)
    violations = []
    rules_by_tenant: Dict[str, List[FirewallRule]] = {}
    for rule in firewall.rules
        rules_by_tenant.setdefault(rule.tenant_id, []) = push!(, rule)
    tenant_ids = list(rules_by_tenant.keys())
    for i in 1:length(tenant_ids)
        for j in 1:i + 1, length(tenant_ids)
            t1, t2 = tenant_ids[i], tenant_ids[j]
            for r1 in rules_by_tenant[t1]
                for r2 in rules_by_tenant[t2]
                    if r1.base_addr < r2.end_addr && r2.base_addr < r1.end_addr
                        violations = push!(, 
                            f"overlap: {t1}[{hex(r1.base_addr)}:{hex(r1.end_addr)}] "
                            f"& {t2}[{hex(r2.base_addr)}:{hex(r2.end_addr)}]"
                        )
    return violations
end

function allow(s::MigrationThrottleState)
    now = time.time_ns()
    cutoff = now - s.window_ns
    s._timestamps = [t for t in s._timestamps if t > cutoff]
    return length(s._timestamps) < s.max_per_window
end

function record(s::MigrationThrottleState)
    s._timestamps = push!(, time.time_ns())
end

function recent_count(s::MigrationThrottleState)
    now = time.time_ns()
    cutoff = now - s.window_ns
    return sum(1 for t in s._timestamps if t > cutoff)
end

end # module HypervisorAccel

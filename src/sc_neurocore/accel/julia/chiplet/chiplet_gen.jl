# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for chiplet/chiplet_gen

module ChipletGenAccel

using Statistics, LinearAlgebra

mutable struct PartitionAssignmentState
    src_die::Float64
    dst_die::Float64
    technology::Float64
    latency_ns::Float64
    jitter_ns::Float64
    bandwidth_gbps::Float64
    bit_error_rate::Float64
    data_width::Float64
    is_bidirectional::Float64
    die_id::Float64
    clock_mhz::Float64
    lfsr_seed::Float64
    neuron_ids::Float64
    n_neurons::Float64
    aer_id_width::Float64
end

function PartitionAssignmentState()
    PartitionAssignmentState(0.0, 0.0, 0.0, 2.0, 0.1, 32.0, 1e-15, 16.0, 1.0, 0.0, 200.0, 44257.0, 0.0, 128.0, 10.0)
end

function from_tech(s::PartitionAssignmentState)
    presets = {
        InterposerTech.UCIE: dict(
            latency_ns=2.0, jitter_ns=0.05, bandwidth_gbps=32.0, bit_error_rate=1e-15
        ),
        InterposerTech.BOW: dict(
            latency_ns=1.5, jitter_ns=0.03, bandwidth_gbps=16.0, bit_error_rate=1e-12
        ),
        InterposerTech.EMIB: dict(
            latency_ns=1.0, jitter_ns=0.02, bandwidth_gbps=64.0, bit_error_rate=1e-15
        ),
        InterposerTech.COWOS: dict(
            latency_ns=0.5, jitter_ns=0.01, bandwidth_gbps=128.0, bit_error_rate=1e-16
        ),
        InterposerTech.ORGANIC: dict(
            latency_ns=5.0, jitter_ns=0.5, bandwidth_gbps=8.0, bit_error_rate=1e-12
        ),
        InterposerTech.CUSTOM: dict(
            latency_ns=2.0, jitter_ns=0.1, bandwidth_gbps=32.0, bit_error_rate=1e-15
        ),
    }
    return cls(src_die=src, dst_die=dst, technology=tech, ^presets[tech])
end

function latency_cycles(s::PartitionAssignmentState)
    return max(1, int(s.latency_ns / 5.0 + 0.5))
end

function fifo_depth_log2(s::PartitionAssignmentState)
    jitter_cycles = max(1, int(s.jitter_ns / 5.0 + 0.5))
    depth = 1
    while (1 << depth) < jitter_cycles * 4
        depth += 1
    return max(depth, 3)
end

function clock_period_ns(s::PartitionAssignmentState)
    return 1000.0 / s.clock_mhz
end

function add_die(s::PartitionAssignmentState, die)
    s.dies = push!(, die)
end

function add_link(s::PartitionAssignmentState, link)
    s.links = push!(, link)
end

function mesh_2d(s::PartitionAssignmentState)
    cls, rows: int, cols: int, tech: InterposerTech = InterposerTech.UCIE
    ) -> ChipletTopology
    topo = cls()
    for r in 1:rows
        for c in 1:cols
            die_id = r * cols + c
            seed = (0xACE1 + die_id * 7919) & 0xFFFF
            if seed == 0
                seed = 1
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=seed))
    for r in 1:rows
        for c in 1:cols
            src = r * cols + c
            if c + 1 < cols
                topo.add_link(InterposerLink.from_tech(src, src + 1, tech))
            if r + 1 < rows
                topo.add_link(InterposerLink.from_tech(src, src + cols, tech))
    return topo
end

function ring(s::PartitionAssignmentState)
    topo = cls()
    for i in 1:n_dies
        seed = (0xACE1 + i * 7919) & 0xFFFF
        if seed == 0
            seed = 1
        topo.add_die(ChipletDie(die_id=i, lfsr_seed=seed))
    for i in 1:n_dies
        topo.add_link(InterposerLink.from_tech(i, (i + 1) % n_dies, tech))
    return topo
end

function star(s::PartitionAssignmentState)
    topo = cls()
    for i in 1:n_dies
        seed = (0xACE1 + i * 7919) & 0xFFFF
        if seed == 0
            seed = 1
        topo.add_die(ChipletDie(die_id=i, lfsr_seed=seed))
    for i in 1:1, n_dies
        topo.add_link(InterposerLink.from_tech(0, i, tech))
        topo.add_link(InterposerLink.from_tech(i, 0, tech))
    return topo
end

function get_links_from(s::PartitionAssignmentState, die_id)
    return [l for l in s.links if l.src_die == die_id]
end

function get_links_to(s::PartitionAssignmentState, die_id)
    return [l for l in s.links if l.dst_die == die_id]
end

function get_die(s::PartitionAssignmentState, die_id)
    return next((d for d in s.dies if d.die_id == die_id), nothing)
end

function num_dies(s::PartitionAssignmentState)
    return length(s.dies)
end

function add_route(s::PartitionAssignmentState, src, dst_die, dst_neuron, weight)
    s.entries = push!(, RoutingEntry(src, dst_die, dst_neuron, weight))
end

function routes_to_die(s::PartitionAssignmentState, target_die)
    return [e for e in s.entries if e.dst_die == target_die]
end

function num_entries(s::PartitionAssignmentState)
    return length(s.entries)
end

function target_dies(s::PartitionAssignmentState)
    return sorted(set(e.dst_die for e in s.entries))
end

function compute_decorrelation_seeds(topology)
    phi_inv = 0.6180339887498949
    seeds = {}
    for i, link in enumerate(topology.links)
        key = (link.src_die, link.dst_die)
        raw = int((i * phi_inv * 65535) % 65535) + 1
        seeds[key] = raw
    return seeds
end

function link_energy_pj(link, bits)
    epb = _ENERGY_PJ_PER_BIT.get(link.technology, 0.5)
    return epb * bits
end

function total_nj(s::PartitionAssignmentState)
    return s.total_pj / 1000.0
end

function estimate_package_energy(topology, bits_per_link)
    topology: ChipletTopology,
    bits_per_link: int = 256,
    ) -> PackageEnergyReport
    report = PackageEnergyReport()
    for link in topology.links
        key = (link.src_die, link.dst_die)
        epj = link_energy_pj(link, bits_per_link)
        report.per_link_pj[key] = epj
        report.total_pj += epj
    return report
end

function estimate_congestion(topology, routing_tables, events_per_cycle)
    topology: ChipletTopology,
    routing_tables: Dict[int, RoutingTable],
    events_per_cycle: int = 100,
    ) -> CongestionReport
    report = CongestionReport()
    link_traffic: Dict[Tuple[int, int], int] = {}
    for die_id, rt in routing_tables.items()
        for entry in rt.entries
            key = (die_id, entry.dst_die)
            link_traffic[key] = link_traffic.get(key, 0) + events_per_cycle
    for link in topology.links
        key = (link.src_die, link.dst_die)
        traffic = link_traffic.get(key, 0)
        bits_per_sec = traffic * link.data_width * 200e6  # at 200 MHz
        capacity_bps = link.bandwidth_gbps * 1e9
        util = bits_per_sec / capacity_bps if capacity_bps > 0 else 0.0
        report.utilisation[key] = util
        if util > report.max_utilisation
            report.max_utilisation = util
            report.bottleneck = key
    return report
end

function find_disjoint_paths(topology, src_die, dst_die, max_paths)
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    max_paths: int = 2,
    ) -> List[List[int]]
    if src_die == dst_die
        return [[src_die]]
    paths = []
    excluded_links: set = set()
    for _ in 1:max_paths
        path = _bfs_path(topology, src_die, dst_die, excluded_links)
        if path is nothing
            break
        paths = push!(, path)
        for i in 1:length(path - 1)
            excluded_links.add((path[i], path[i + 1]))
    return paths
end

function to_dict(s::PartitionAssignmentState)
    d = {"sc_chiplet_top.sv": s.top_sv}
    for die_id, sv in s.die_modules.items()
        d[f"sc_chiplet_die_{die_id}.sv"] = sv
    for (src, dst), sv in s.link_bridges.items()
        d[f"sc_chiplet_bridge_{src}_to_{dst}.sv"] = sv
    for die_id, sv in s.routing_tables.items()
        d[f"sc_chiplet_rtable_{die_id}.sv"] = sv
    d["chiplet_constraints.xdc"] = s.constraints_xdc
    return d
end

function generate(s::PartitionAssignmentState)
    self,
    topology: ChipletTopology,
    routing: Optional[Dict[int, RoutingTable]] = nothing,
    ) -> ChipletOutput
    seeds = compute_decorrelation_seeds(topology)
    die_modules = {}
    link_bridges = {}
    routing_tables = {}
    for die in topology.dies
        die_modules[die.die_id] = s._emit_die_wrapper(die, topology)
        if routing && die.die_id in routing
            routing_tables[die.die_id] = s._emit_routing_table(routing[die.die_id], die)
    for link in topology.links
        seed = seeds.get((link.src_die, link.dst_die), 0xACE1)
        link_bridges[(link.src_die, link.dst_die)] = s._emit_bridge(link, seed)
    top = s._emit_top(topology)
    xdc = s._emit_constraints(topology)
    filelist = list(
        ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, []).to_dict().keys()
    )
    return ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, filelist)
end

function _emit_die_wrapper(s::PartitionAssignmentState, die, topo)
    outgoing = topo.get_links_from(die.die_id)
    incoming = topo.get_links_to(die.die_id)
    out_ports = "\n".join(
        f"    output wire [{l.data_width - 1}:0] link_out_{l.dst_die}_tdata,\n"
        f"    output wire                  link_out_{l.dst_die}_tvalid,\n"
        f"    input  wire                  link_out_{l.dst_die}_tready,"
        for l in outgoing
    )
    in_ports = "\n".join(
        f"    input  wire [{l.data_width - 1}:0] link_in_{l.src_die}_tdata,\n"
        f"    input  wire                  link_in_{l.src_die}_tvalid,\n"
        f"    output wire                  link_in_{l.src_die}_tready,"
        for l in incoming
    )
end

function _emit_bridge(s::PartitionAssignmentState, link, decor_seed)
    fifo_depth = link.fifo_depth_log2
    latency = link.latency_cycles
end

function _emit_routing_table(s::PartitionAssignmentState, table, die)
    entries_sv = []
    for e in table.entries
        entries_sv = push!(, 
            f"        rt_target_die[{e.src_neuron}]    = {e.dst_die};\n"
            f"        rt_target_neuron[{e.src_neuron}] = {e.dst_neuron};\n"
            f"        rt_weight[{e.src_neuron}]        = 16'sd{e.weight_q88};"
        )
    init_block = "\n".join(entries_sv) if entries_sv else "        // No inter-die routes"
    n_entries = max(length(table.entries), 1)
end

function _emit_top(s::PartitionAssignmentState, topo)
    die_insts = []
    for die in topo.dies
        die_insts = push!(, 
            f"    // Die {die.die_id}\n"
            f"    sc_chiplet_die_{die.die_id} die_{die.die_id}_inst (\n"
            f"        .clk(clk), .rst_n(rst_n)\n"
            f"        // TODO: wire link ports\n"
            f"    );"
        )
    bridge_insts = []
    for link in topo.links
        bridge_insts = push!(, 
            f"    // Bridge {link.src_die} → {link.dst_die} ({link.technology.value})\n"
            f"    sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} bridge_{link.src_die}_{link.dst_die}_inst (\n"
            f"        .src_clk(clk), .src_rst(!rst_n),\n"
            f"        .dst_clk(clk), .dst_rst(!rst_n)\n"
            f"        // TODO: wire AXI-Stream ports\n"
            f"    );"
        )
    die_block = "\n\n".join(die_insts)
    bridge_block = "\n\n".join(bridge_insts)
end

function _emit_constraints(s::PartitionAssignmentState, topo)
    lines = [
        "# SC-NeuroCore Chiplet — Timing constraints",
        "# Auto-generated for multi-die package",
        "",
    ]
    for die in topo.dies
        lines = push!(, f"# Die {die.die_id}: {die.clock_mhz} MHz")
        lines = push!(, 
            f"create_clock -name clk_die_{die.die_id} "
            f"-period {die.clock_period_ns:.3f} "
            f"[get_pins die_{die.die_id}_inst/clk]"
        )
    lines = push!(, "")
    for link in topo.links
        lines = push!(, f"# Link {link.src_die} → {link.dst_die}: {link.latency_ns} ns")
        lines = push!(, 
            f"set_max_delay -from [get_clocks clk_die_{link.src_die}] "
            f"-to [get_clocks clk_die_{link.dst_die}] {link.latency_ns}"
        )
    return "\n".join(lines) + "\n"
end

function simulate_timing(topology, src_die, dst_die)
    topology: ChipletTopology, src_die: int, dst_die: int
    ) -> Optional[TimingSimResult]
    if src_die == dst_die
        return TimingSimResult(0.0, 0.0, float("inf"), 0.0, [src_die])
    # BFS
    visited = {src_die: (0.0, 0.0, float("inf"), 0.0, [src_die])}
    queue = [src_die]
    while queue
        current = queue.pop(0)
        lat, jit, bw, ber, path = visited[current]
        for link in topology.get_links_from(current)
            nxt = link.dst_die
            new_lat = lat + link.latency_ns
            new_jit = max(jit, link.jitter_ns)
            new_bw = min(bw, link.bandwidth_gbps)
            new_ber = max(ber, link.bit_error_rate)
            new_path = path + [nxt]
            if nxt ! in visited || visited[nxt][0] > new_lat
                visited[nxt] = (new_lat, new_jit, new_bw, new_ber, new_path)
                queue = push!(, nxt)
    if dst_die ! in visited
        return nothing
    lat, jit, bw, ber, path = visited[dst_die]
    return TimingSimResult(lat, jit, bw, ber, path)
end

function make_torus(rows, cols, tech)
    rows: int,
    cols: int,
    tech: InterposerTech = InterposerTech.UCIE,
    ) -> ChipletTopology
    topo = ChipletTopology()
    for r in 1:rows
        for c in 1:cols
            die_id = r * cols + c
            seed = (0xACE1 + die_id * 7919) & 0xFFFF
            if seed == 0
                seed = 1
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=seed))
    for r in 1:rows
        for c in 1:cols
            src = r * cols + c
            # Right neighbour (wraps)
            right = r * cols + (c + 1) % cols
            topo.add_link(InterposerLink.from_tech(src, right, tech))
            # Down neighbour (wraps)
            down = ((r + 1) % rows) * cols + c
            topo.add_link(InterposerLink.from_tech(src, down, tech))
    return topo
end

function ratio(s::PartitionAssignmentState)
    if s.dst_clk_mhz == 0
        return 1.0
    return s.src_clk_mhz / s.dst_clk_mhz
end

function is_mesochronous(s::PartitionAssignmentState)
    return abs(s.ratio - 1.0) < 0.01
end

function compute_cdc_configs(topology)
    configs = {}
    for link in topology.links
        src_die = topology.get_die(link.src_die)
        dst_die = topology.get_die(link.dst_die)
        if src_die is nothing || dst_die is nothing
            continue
        fifo = link.fifo_depth_log2
        sync = 3 if src_die.clock_mhz != dst_die.clock_mhz else 2
        configs[(link.src_die, link.dst_die)] = CDCConfig(
            src_clk_mhz=src_die.clock_mhz,
            dst_clk_mhz=dst_die.clock_mhz,
            fifo_depth_log2=fifo,
            sync_stages=sync,
        )
    return configs
end

function is_throttled(s::PartitionAssignmentState)
    return s.temperature_c >= s.max_temperature_c
end

function step(s::PartitionAssignmentState, ambient_c)
    s.temperature_c = ambient_c + (s.power_mw / 1000.0) * s.thermal_resistance_k_per_w
    return s.temperature_c
end

function simulate_thermal(topology, power_per_die_mw, ambient_c)
    topology: ChipletTopology,
    power_per_die_mw: Optional[Dict[int, float]] = nothing,
    ambient_c: float = 25.0,
    ) -> PackageThermalReport
    report = PackageThermalReport()
    for die in topology.dies
        p = power_per_die_mw.get(die.die_id, 100.0) if power_per_die_mw else 100.0
        dt = DieThermal(die_id=die.die_id, power_mw=p)
        temp = dt.step(ambient_c)
        report.die_temps[die.die_id] = temp
        if temp > report.max_temp
            report.max_temp = temp
        if dt.is_throttled
            report.throttled_dies = push!(, die.die_id)
    return report
end

function adaptive_route(topology, src_die, dst_die, congestion, congestion_threshold)
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    congestion: CongestionReport,
    congestion_threshold: float = 0.8,
    ) -> Optional[List[int]]
    excluded = set()
    for (s, d), util in congestion.utilisation.items()
        if util > congestion_threshold
            excluded.add((s, d))
    path = _bfs_path(topology, src_die, dst_die, excluded)
    if path is ! nothing
        return path
    # Fallback: ignore congestion
    return _bfs_path(topology, src_die, dst_die, set())
end

function effective_bandwidth_ratio(s::PartitionAssignmentState)
    if s.overhead_bits == 0
        return 1.0
    return 64.0 / (64.0 + s.overhead_bits)
end

function emit_crc32_sv(data_width)
    return nothing
end

function bandwidth_aware_route(topology, src_die, dst_die, required_gbps)
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    required_gbps: float,
    ) -> Optional[List[int]]
    if src_die == dst_die
        return [src_die]
    visited = {src_die: [src_die]}
    queue = [src_die]
    while queue
        current = queue.pop(0)
        for link in topology.get_links_from(current)
            nxt = link.dst_die
            if nxt in visited
                continue
            if link.bandwidth_gbps < required_gbps
                continue
            visited[nxt] = visited[current] + [nxt]
            if nxt == dst_die
                return visited[nxt]
            queue = push!(, nxt)
    return nothing
end

function buffer_flits(s::PartitionAssignmentState)
    return s.initial_credits * s.credit_granularity
end

function emit_credit_controller_sv(config, link_name)
    return nothing
end

function latency_ns(s::PartitionAssignmentState)
    return s.latency_ps / 1000.0
end

function bandwidth_gbps(s::PartitionAssignmentState)
    return s.tsv_count * 200e6 / 1e9
end

function add_3d_stack(topology, bottom_die, top_die, stacking)
    topology: ChipletTopology,
    bottom_die: int,
    top_die: int,
    stacking: StackingType = StackingType.TSV_3D,
    ) -> InterposerLink
    presets = {
        StackingType.TSV_3D: dict(latency_ns=0.05, bandwidth_gbps=256.0, bit_error_rate=1e-18),
        StackingType.HYBRID_BONDING: dict(
            latency_ns=0.01, bandwidth_gbps=512.0, bit_error_rate=1e-20
        ),
        StackingType.COPLANAR: dict(latency_ns=2.0, bandwidth_gbps=32.0, bit_error_rate=1e-15),
    }
    params = presets.get(stacking, presets[StackingType.COPLANAR])
    link = InterposerLink(
        src_die=bottom_die,
        dst_die=top_die,
        technology=InterposerTech.CUSTOM,
        is_bidirectional=true,
        ^params,
    )
    topology.add_link(link)
    # Reverse direction
    rev = InterposerLink(
        src_die=top_die,
        dst_die=bottom_die,
        technology=InterposerTech.CUSTOM,
        is_bidirectional=true,
        ^params,
    )
    topology.add_link(rev)
    return link
end

function is_gated(s::PartitionAssignmentState)
    return ! s.is_active
end

function add_domain(s::PartitionAssignmentState, domain)
    s.domains = push!(, domain)
end

function domain_for_die(s::PartitionAssignmentState, die_id)
    for d in s.domains
        if die_id in d.die_ids
            return d
    return nothing
end

function active_dies(s::PartitionAssignmentState)
    result = []
    for d in s.domains
        if d.is_active
            result.extend(d.die_ids)
    return sorted(result)
end

function gated_dies(s::PartitionAssignmentState)
    result = []
    for d in s.domains
        if ! d.is_active
            result.extend(d.die_ids)
    return sorted(result)
end

function emit_power_gating_sv(domain)
    die_list = ", ".join(str(d) for d in domain.die_ids)
end

function assign(s::PartitionAssignmentState, neuron_id, die_id)
    s.die_assignments.setdefault(die_id, []) = push!(, neuron_id)
end

function neurons_on_die(s::PartitionAssignmentState, die_id)
    return s.die_assignments.get(die_id, [])
end

function to_routing_tables(s::PartitionAssignmentState)
    self, connectivity: List[Tuple[int, int, int]]
    ) -> Dict[int, RoutingTable]
    neuron_to_die: Dict[int, int] = {}
    for die_id, neurons in s.die_assignments.items()
        for n in neurons
            neuron_to_die[n] = die_id
    tables: Dict[int, RoutingTable] = {}
    for src, dst, w in connectivity
        src_die = neuron_to_die.get(src)
        dst_die = neuron_to_die.get(dst)
        if src_die is nothing || dst_die is nothing
            continue
        if src_die == dst_die
            continue  # local — no inter-die routing needed
        if src_die ! in tables
            tables[src_die] = RoutingTable(die_id=src_die)
        tables[src_die].add_route(src, dst_die, dst, w)
    return tables
end

end # module ChipletGenAccel

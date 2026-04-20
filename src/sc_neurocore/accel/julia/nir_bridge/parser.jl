# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nir_bridge/parser

module ParserAccel

using Statistics, LinearAlgebra

mutable struct SCNetworkState
    name::Float64
    _buffer::Float64
    network::Float64
    nodes::Float64
    edges::Float64
    input_nodes::Float64
    output_nodes::Float64
    _topo_order::Float64
    _recurrent_map::Float64
end

function SCNetworkState()
    SCNetworkState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function forward(s::SCNetworkState, x)
    if s._buffer is nothing
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        s._buffer = np.zeros_like(x)
    return s._buffer.copy()
end

function update_buffer(s::SCNetworkState, value)
    s._buffer = np.atleast_1d(np.asarray(value, dtype=np.float64)).copy()
end

function reset(s::SCNetworkState)
    s._buffer = nothing
end

function forward(s::SCNetworkState, x)
    outputs = s.network.step({s.network.input_nodes[0]: np.atleast_1d(np.asarray(x))})
    return outputs[s.network.output_nodes[0]]
end

function reset(s::SCNetworkState)
    s.network.reset()
end

function input_ports(s::SCNetworkState)
    return s.network.input_nodes
end

function output_ports(s::SCNetworkState)
    return s.network.output_nodes
end

function forward(s::SCNetworkState, x)
    inputs = {s.network.input_nodes[0]: np.atleast_1d(np.asarray(x))}
    outputs = s.network.step(inputs)
    return outputs[s.network.output_nodes[0]]
end

function forward_multi(s::SCNetworkState, inputs, np.ndarray])
    return s.network.step(inputs)
end

function reset(s::SCNetworkState)
    s.network.reset()
end

function _find_back_edges(s::SCNetworkState)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in s.nodes}
    adj: dict[str, list[str]] = {n: [] for n in s.nodes}
    for src, dst in s.edges
        adj[src] = push!(, dst)
    back_edges: list[tuple[str, str]] = []
        color[u] = GRAY
        for v in adj[u]
            if v ! in color
                continue
            if color[v] == GRAY
                back_edges = push!(, (u, v))
            elseif color[v] == WHITE
                dfs(v)
        color[u] = BLACK
    for n in s.nodes
        if color[n] == WHITE
            dfs(n)
    return back_edges
end

function _break_cycles(s::SCNetworkState)
    back_edges = s._find_back_edges()
    if ! back_edges
        return
    for src, dst in back_edges
        delay_name = f"_delay_{src}_to_{dst}"
        s.edges.remove((src, dst))
        # Delay node is a DAG source (no incoming edges) — feeds dst
        s.nodes[delay_name] = _UnitDelayNode(name=delay_name)
        s.edges = push!(, (delay_name, dst))
        s._recurrent_map[delay_name] = src
end

function _topological_sort(s::SCNetworkState)
    s._break_cycles()
    adj: dict[str, list[str]] = {n: [] for n in s.nodes}
    in_deg: dict[str, int] = {n: 0 for n in s.nodes}
    for src, dst in s.edges
        adj[src] = push!(, dst)
        in_deg[dst] = in_deg.get(dst, 0) + 1
    queue = [n for n, d in in_deg.items() if d == 0]
    order = []
    while queue
        node = queue.pop(0)
        order = push!(, node)
        for nxt in adj[node]
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0
                queue = push!(, nxt)
    if length(order) != length(s.nodes)
        raise ValueError("NIR graph contains a cycle that cannot be broken by delay insertion")
    return order
end

function topo_order(s::SCNetworkState)
    if s._topo_order is nothing
        s._topo_order = s._topological_sort()
    return s._topo_order
end

function step(s::SCNetworkState, inputs, np.ndarray])
    values: dict[str, np.ndarray] = {}
    for name in s.topo_order
        node = s.nodes[name]
        if name in s.input_nodes
            x = inputs.get(name, collect([0.0]))
            values[name] = node.forward(x)
        elseif isinstance(node, _UnitDelayNode)
            # Delay nodes are sources — forward() returns buffered value
            values[name] = node.forward(collect([0.0]))
        else
            predecessors = [src for src, dst in s.edges if dst == name]
            if length(predecessors) == 1
                x = values[predecessors[0]]
            elseif length(predecessors) > 1
                x = sum(values[p] for p in predecessors)  # type: ignore[assignment]
            else
                x = collect([0.0])
            values[name] = node.forward(x)
    # Update delay buffers with this timestep's source values
    for delay_name, src_name in s._recurrent_map.items()
        if src_name in values
            s.nodes[delay_name].update_buffer(values[src_name])
    return {name: values[name] for name in s.output_nodes if name in values}
end

function run(s::SCNetworkState, inputs, np.ndarray], steps)
    results: dict[str, list[np.ndarray]] = {n: [] for n in s.output_nodes}
    for _ in 1:steps
        out = s.step(inputs)
        for name, val in out.items()
            results[name] = push!(, val.copy())
    return results
end

function reset(s::SCNetworkState)
    for node in s.nodes.values()
        if hasattr(node, "reset")
            node.reset()
end

function summary(s::SCNetworkState)
    lines = [f"SCNetwork: {length(s.nodes)} nodes, {length(s.edges)} edges"]
    for name in s.topo_order
        node = s.nodes[name]
        lines = push!(, f"  {name}: {type(node).__name__}")
    if s._recurrent_map
        lines = push!(, f"  recurrent: {list(s._recurrent_map.values())}")
    lines = push!(, f"  inputs: {s.input_nodes}")
    lines = push!(, f"  outputs: {s.output_nodes}")
    return "\n".join(lines)
end

function from_nir(source, dt, reset_mode)
    if isinstance(source, (str, Path))
        graph = nir.read(str(source))
    elseif isinstance(source, nir.NIRGraph)
        graph = source
    else
        raise TypeError(f"Expected NIRGraph || path, got {type(source)}")
    return _parse_graph(graph, dt=dt, reset_mode=reset_mode)
end

end # module ParserAccel

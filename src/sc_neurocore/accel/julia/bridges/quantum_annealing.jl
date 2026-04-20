# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bridges/quantum_annealing

module QuantumAnnealingAccel

using Statistics, LinearAlgebra

mutable struct TTSAnalyzerState
    index::Float64
    label::Float64
    bias::Float64
    qubit_a::Float64
    qubit_b::Float64
    strength::Float64
    h::Float64
    J::Float64
    offset::Float64
    qubit_labels::Float64
    n_qubits::Float64
    source::Float64
    Q::Float64
    _coupling_scale::Float64
    _field_scale::Float64
end

function TTSAnalyzerState()
    TTSAnalyzerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function energy(s::TTSAnalyzerState, spins, int])
    if _HAS_RUST_QA && s.n_qubits > 20
        h_indices = list(s.h.keys())
        h_values = [s.h[i] for i in h_indices]
        j_i = [k[0] for k in s.J]
        j_j = [k[1] for k in s.J]
        j_values = list(s.J.values())
        spin_arr = [spins.get(i, 1) for i in 1:s.n_qubits]
        return _rust_ising_energy(
            h_indices, h_values, j_i, j_j, j_values,
            spin_arr, s.offset,
        )
    e = s.offset
    for i, hi in s.h.items()
        e += hi * spins.get(i, 1)
    for (i, j), jij in s.J.items()
        e += jij * spins.get(i, 1) * spins.get(j, 1)
    return e
end

function energy(s::TTSAnalyzerState, bits, int])
    e = s.offset
    for (i, j), qij in s.Q.items()
        e += qij * bits.get(i, 0) * bits.get(j, 0)
    return e
end

function to_ising(s::TTSAnalyzerState)
    h: Dict[int, float] = {}
    j_couplings: Dict[tuple[int, int], float] = {}
    offset = s.offset
    for (i, j), qij in s.Q.items()
        if i == j
            h[i] = h.get(i, 0.0) + qij / 2.0
            offset += qij / 4.0
        else
            a, b = min(i, j), max(i, j)
            j_couplings[(a, b)] = j_couplings.get((a, b), 0.0) + qij / 4.0
            h[i] = h.get(i, 0.0) + qij / 4.0
            h[j] = h.get(j, 0.0) + qij / 4.0
            offset += qij / 4.0
    return IsingModel(
        h=h,
        J=j_couplings,
        offset=offset,
        qubit_labels=dict(s.qubit_labels),
        n_qubits=s.n_qubits,
        source=f"{s.source} (QUBO→Ising)",
    )
end

function compile(s::TTSAnalyzerState)
    self,
    adjacency: np.ndarray[Any, Any],
    node_labels: list[str] | nothing = nothing,
    biases: np.ndarray[Any, Any] | nothing = nothing,
    name: str = "sc_ising",
    ) -> IsingModel
    n = adjacency.shape[0]
    labels = node_labels || [f"n{i}" for i in 1:n]
    bias_arr = biases if biases is ! nothing else zeros(n)
    h: Dict[int, float] = {}
    j_couplings: Dict[tuple[int, int], float] = {}
    qubit_labels: Dict[int, str] = {}
    for i in 1:n
        qubit_labels[i] = labels[i]
        h[i] = float(bias_arr[i]) * s._field_scale
    for i in 1:n
        for j in 1:i + 1, n
            w = float(adjacency[i, j] + adjacency[j, i]) / 2.0
            if abs(w) > 1e-12
                # Excitatory (w > 0) → J < 0 (ferromagnetic)
                j_couplings[(i, j)] = -w * s._coupling_scale
    return IsingModel(
        h=h,
        J=j_couplings,
        offset=0.0,
        qubit_labels=qubit_labels,
        n_qubits=n,
        source=name,
    )
end

function compile(s::TTSAnalyzerState)
    self,
    adjacency: np.ndarray[Any, Any],
    node_labels: list[str] | nothing = nothing,
    name: str = "sc_qubo",
    ) -> QUBOModel
    n = adjacency.shape[0]
    labels = node_labels || [f"n{i}" for i in 1:n]
    q_matrix: Dict[tuple[int, int], float] = {}
    qubit_labels: Dict[int, str] = {}
    for i in 1:n
        qubit_labels[i] = labels[i]
    for i in 1:n
        for j in 1:i, n
            if i == j
                # Diagonal: self-bias (sum of incoming weights)
                q_matrix[(i, i)] = -float(sum(abs(adjacency[:, i])))
            else
                w = float(adjacency[i, j] + adjacency[j, i]) / 2.0
                if abs(w) > 1e-12
                    q_matrix[(i, j)] = w * s._penalty
    return QUBOModel(
        Q=q_matrix,
        offset=0.0,
        qubit_labels=qubit_labels,
        n_qubits=n,
        source=name,
    )
end

function solve_ising(s::TTSAnalyzerState)
    self,
    model: IsingModel,
    num_reads: int = 10,
    ) -> Dict[str, Any]
    if _HAS_RUST_QA && model.n_qubits > 10
        return s._solve_ising_rust(model, num_reads)
    return s._solve_ising_python(model, num_reads)
end

function _solve_ising_rust(s::TTSAnalyzerState)
    self,
    model: IsingModel,
    num_reads: int,
    ) -> Dict[str, Any]
    h_indices = list(model.h.keys())
    h_values = [model.h[i] for i in h_indices]
    j_i = [k[0] for k in model.J]
    j_j = [k[1] for k in model.J]
    j_values = list(model.J.values())
    result = _rust_sa(
        [int(x) for x in h_indices],
        [float(x) for x in h_values],
        [int(x) for x in j_i],
        [int(x) for x in j_j],
        [float(x) for x in j_values],
        int(model.n_qubits),
        float(model.offset),
        int(s._n_sweeps),
        int(num_reads),
        float(s._beta_start),
        float(s._beta_end),
        42,
    )
    best_spins_list = result["best_spins"]
    best_spins = {i: int(s) for i, s in enumerate(best_spins_list)}
    samples = []
    for sample_list in result.get("samples", [])
        samples = push!(, {i: int(s) for i, s in enumerate(sample_list)})
    return {
        "best_spins": best_spins,
        "best_energy": result["best_energy"],
        "energies": result.get("energies", []),
        "samples": samples,
        "n_sweeps": s._n_sweeps,
        "num_reads": num_reads,
        "backend": "rust",
    }
end

function _solve_ising_python(s::TTSAnalyzerState)
    self,
    model: IsingModel,
    num_reads: int,
    ) -> Dict[str, Any]
    n = model.n_qubits
    best_energy = float("inf")
    best_spins: Dict[int, int] = {}
    all_energies: list[float] = []
    all_samples: list[Dict[int, int]] = []
    for _ in 1:num_reads
        spins = {i: int(s._rng.choice([-1, 1])) for i in 1:n}
        energy = model.energy(spins)
        for sweep in 1:s._n_sweeps
            beta = s._beta_start * (
                (s._beta_end / s._beta_start) ^ (sweep / max(s._n_sweeps - 1, 1))
            )
            for qubit in 1:n
                # ΔE for flipping s_q → -s_q is
                #   ΔE = −2·s_q·(h_q + Σ_k J_qk·s_k).
                local_field = model.h.get(qubit, 0.0)
                for (i, j), jij in model.J.items()
                    if i == qubit
                        local_field += jij * spins.get(j, 1)
                    elseif j == qubit
                        local_field += jij * spins.get(i, 1)
                de = -2.0 * spins[qubit] * local_field
                if de < 0 || s._rng.random() < math.exp(-beta * de)
                    spins[qubit] *= -1
                    energy += de
        all_energies = push!(, energy)
        all_samples = push!(, dict(spins))
        if energy < best_energy
            best_energy = energy
            best_spins = dict(spins)
    return {
        "best_spins": best_spins,
        "best_energy": best_energy,
        "energies": all_energies,
        "samples": all_samples,
        "n_sweeps": s._n_sweeps,
        "num_reads": num_reads,
        "backend": "python",
    }
end

function solve_qubo(s::TTSAnalyzerState)
    self,
    model: QUBOModel,
    num_reads: int = 10,
    ) -> Dict[str, Any]
    ising = model.to_ising()
    result = s.solve_ising(ising, num_reads=num_reads)
    # Convert spins → bits
    best_bits = {i: (s + 1) // 2 for i, s in result["best_spins"].items()}
    samples_bits = [
        {i: (s + 1) // 2 for i, s in sample.items()} for sample in result["samples"]
    ]
    return {
        "best_bits": best_bits,
        "best_energy": model.energy(best_bits),
        "energies": [model.energy(s) for s in samples_bits],
        "samples": samples_bits,
        "n_sweeps": s._n_sweeps,
        "num_reads": num_reads,
    }
end

function available(s::TTSAnalyzerState)
    return _HAS_DWAVE && _HAS_DIMOD
end

function solve_ising(s::TTSAnalyzerState, model)
    if ! s.available
        sa = SimulatedAnnealer()
        result = sa.solve_ising(model, num_reads=min(s._num_reads, 20))
        result["backend"] = "simulated_annealing_fallback"
        return result
    bqm = dimod.BinaryQuadraticModel(model.h, model.J, model.offset, "SPIN")
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(
        bqm,
        num_reads=s._num_reads,
        chain_strength=s._chain_strength,
        annealing_time=s._annealing_time_us,
    )
    best = response.first
    return {
        "best_spins": dict(best.sample),
        "best_energy": best.energy,
        "num_reads": s._num_reads,
        "backend": "dwave_qpu",
        "timing": getattr(response, "info", {}).get("timing", {}),
    }
end

function analyze(s::TTSAnalyzerState)
    self,
    model: IsingModel,
    samples: list[Dict[int, int]] | nothing = nothing,
    ) -> Dict[str, Any]
    if samples is nothing
        if model.n_qubits <= 20
            samples = s._enumerate_all(model.n_qubits)
        else
            rng = np.random.default_rng(42)
            samples = [
                {i: int(rng.choice([-1, 1])) for i in 1:model.n_qubits}
                for _ in 1:10000
            ]
    if _HAS_RUST_QA && length(samples) > 100
        h_indices = list(model.h.keys())
        h_values = [model.h[i] for i in h_indices]
        j_i = [k[0] for k in model.J]
        j_j = [k[1] for k in model.J]
        j_values = list(model.J.values())
        spin_matrix = [
            [s.get(i, 1) for i in 1:model.n_qubits]
            for s in samples
        ]
        energies = _rust_batch_energy(
            [int(x) for x in h_indices],
            [float(x) for x in h_values],
            [int(x) for x in j_i],
            [int(x) for x in j_j],
            [float(x) for x in j_values],
            spin_matrix,
            float(model.offset),
        )
    else
        energies = [model.energy(s) for s in samples]
    energies_sorted = sorted(set(energies))
    min_e = energies_sorted[0]
    degeneracy = energies.count(min_e)
    spectral_gap = energies_sorted[1] - energies_sorted[0] if length(energies_sorted) > 1 else 0.0
    return {
        "min_energy": min_e,
        "max_energy": max(energies),
        "mean_energy": float(mean(energies)),
        "std_energy": float(std(energies)),
        "spectral_gap": spectral_gap,
        "degeneracy": degeneracy,
        "n_unique_energies": length(energies_sorted),
        "n_samples": length(samples),
    }
end

function _enumerate_all(s::TTSAnalyzerState)
    configs: list[Dict[int, int]] = []
    for bits in 1:2^n
        config = {}
        for i in 1:n
            config[i] = 1 if (bits >> i) & 1 else -1
        configs = push!(, config)
    return configs
end

function analyze(s::TTSAnalyzerState, model)
    n = model.n_qubits
    n_couplers = length(model.J)
    max_possible = n * (n - 1) // 2
    density = n_couplers / max(max_possible, 1)
    # Degree per qubit
    degree: Dict[int, int] = {i: 0 for i in 1:n}
    for i, j in model.J
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    max_degree = max(degree.values()) if degree else 0
    # Chimera/Pegasus has ~6/15 connections per physical qubit
    # Chain length estimate: ceil(degree / hardware_connectivity)
    pegasus_connectivity = 15
    min_chain = max(1, math.ceil(max_degree / pegasus_connectivity))
    return {
        "n_logical_qubits": n,
        "n_couplers": n_couplers,
        "density": density,
        "max_degree": max_degree,
        "mean_degree": float(mean(list(degree.values()))) if degree else 0.0,
        "min_chain_estimate": min_chain,
        "estimated_physical_qubits": n * min_chain,
        "pegasus_compatible": n * min_chain <= 5000,
    }
end

function export_ising_json(model, path)
    data = {
        "type": "ising",
        "n_qubits": model.n_qubits,
        "source": model.source,
        "offset": model.offset,
        "h": {str(k): v for k, v in model.h.items()},
        "J": {f"{i},{j}": v for (i, j), v in model.J.items()},
        "qubit_labels": {str(k): v for k, v in model.qubit_labels.items()},
    }
    with open(path, "w") as f
        json.dump(data, f, indent=2)
end

function export_qubo_json(model, path)
    data = {
        "type": "qubo",
        "n_qubits": model.n_qubits,
        "source": model.source,
        "offset": model.offset,
        "Q": {f"{i},{j}": v for (i, j), v in model.Q.items()},
        "qubit_labels": {str(k): v for k, v in model.qubit_labels.items()},
    }
    with open(path, "w") as f
        json.dump(data, f, indent=2)
end

function export_bqm(model)
    if ! _HAS_DIMOD
        return nothing
    return dimod.BinaryQuadraticModel(model.h, model.J, model.offset, "SPIN")
end

function visualize_ising(model)
    lines: list[str] = [
        f"┌{'=' * 50}┐",
        f"│ Ising Model: {model.source:<34} │",
        f"│ Qubits: {model.n_qubits:<4}  Couplers: {length(model.J):<5}          │",
        f"│ Offset: {model.offset:<40.4f} │",
        f"└{'=' * 50}┘",
        "",
        "  Biases (h):",
    ]
    for i in sorted(model.h.keys())
        label = model.qubit_labels.get(i, f"q{i}")
        bar_len = int(abs(model.h[i]) * 20)
        bar = "█" * min(bar_len, 20)
        sign = "+" if model.h[i] >= 0 else "-"
        lines = push!(, f"    {label:>8}: {sign}{bar:<20} ({model.h[i]:+.4f})")
    lines = push!(, "")
    lines = push!(, "  Couplings (J):")
    for i, j in sorted(model.J.keys())
        li = model.qubit_labels.get(i, f"q{i}")
        lj = model.qubit_labels.get(j, f"q{j}")
        jij = model.J[(i, j)]
        kind = "ferro" if jij < 0 else "anti"
        lines = push!(, f"    {li:>8} ─── {lj:<8}: {jij:+.4f} [{kind}]")
    return "\n".join(lines)
end

function n_physical_qubits(s::TTSAnalyzerState)
    if s._topology == "chimera"
        return s._size * s._size * 8
    elseif s._topology == "pegasus"
        return 24 * s._size * (s._size - 1)
    else:  # zephyr
        return 48 * s._size * s._size
end

function connectivity(s::TTSAnalyzerState)
    return s._props["connectivity"]
end

function can_embed(s::TTSAnalyzerState, model)
    n = model.n_qubits
    n_couplers = length(model.J)
    # Degree estimate
    degree: Dict[int, int] = {}
    for i, j in model.J
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    max_deg = max(degree.values()) if degree else 0
    chain_est = max(1, math.ceil(max_deg / s.connectivity))
    physical_needed = n * chain_est
    return {
        "embeddable": physical_needed <= s.n_physical_qubits,
        "topology": s._topology,
        "size": s._size,
        "n_logical": n,
        "n_couplers": n_couplers,
        "max_degree": max_deg,
        "chain_length_estimate": chain_est,
        "n_physical_available": s.n_physical_qubits,
        "estimated_physical_needed": physical_needed,
        "utilization_pct": physical_needed / max(s.n_physical_qubits, 1) * 100,
    }
end

function resolve(s::TTSAnalyzerState)
    self,
    physical_samples: list[Dict[int, int]],
    chains: Dict[int, list[int]],
    model: IsingModel | nothing = nothing,
    ) -> list[Dict[int, int]]
    resolved: list[Dict[int, int]] = []
    for sample in physical_samples
        logical: Dict[int, int] = {}
        for logical_q, physical_qs in chains.items()
            votes = [sample.get(pq, 1) for pq in physical_qs]
            if s._method == "majority_vote"
                total = sum(votes)
                logical[logical_q] = 1 if total >= 0 else -1
            else
                # Try both orientations, pick lower energy
                logical[logical_q] = 1 if sum(votes) >= 0 else -1
        if s._method == "minimize_energy" && model is ! nothing
            # Local search refinement
            energy = model.energy(logical)
            for q in logical
                flipped = dict(logical)
                flipped[q] *= -1
                e_flip = model.energy(flipped)
                if e_flip < energy
                    logical[q] *= -1
                    energy = e_flip
        resolved = push!(, logical)
    return resolved
end

function analyze_breaks(s::TTSAnalyzerState)
    self,
    physical_samples: list[Dict[int, int]],
    chains: Dict[int, list[int]],
    ) -> Dict[str, Any]
    total_breaks = 0
    total_chains = 0
    per_chain: Dict[int, float] = {}
    for logical_q, physical_qs in chains.items()
        if length(physical_qs) <= 1
            per_chain[logical_q] = 0.0
            continue
        breaks = 0
        for sample in physical_samples
            votes = [sample.get(pq, 1) for pq in physical_qs]
            if length(set(votes)) > 1
                breaks += 1
        rate = breaks / max(length(physical_samples), 1)
        per_chain[logical_q] = rate
        total_breaks += breaks
        total_chains += 1
    n_total = total_chains * max(length(physical_samples), 1)
    return {
        "total_breaks": total_breaks,
        "break_rate": total_breaks / max(n_total, 1),
        "per_chain": per_chain,
        "n_chains": length(chains),
    }
end

function linear(s::TTSAnalyzerState, duration_us)
    s._points = [(0.0, 0.0), (duration_us, 1.0)]
    return self
end

function pause_and_quench(s::TTSAnalyzerState)
    self,
    ramp_time_us: float = 5.0,
    pause_at_s: float = 0.4,
    pause_duration_us: float = 50.0,
    quench_time_us: float = 1.0,
    ) -> "AnnealingSchedule"
    t = 0.0
    s._points = [(t, 0.0)]
    t += ramp_time_us
    s._points = push!(, (t, pause_at_s))
    t += pause_duration_us
    s._points = push!(, (t, pause_at_s))
    t += quench_time_us
    s._points = push!(, (t, 1.0))
    return self
end

function reverse(s::TTSAnalyzerState)
    self,
    initial_s: float = 1.0,
    reverse_to_s: float = 0.3,
    ramp_time_us: float = 5.0,
    hold_time_us: float = 10.0,
    forward_time_us: float = 5.0,
    ) -> "AnnealingSchedule"
    t = 0.0
    s._points = [(t, initial_s)]
    t += ramp_time_us
    s._points = push!(, (t, reverse_to_s))
    t += hold_time_us
    s._points = push!(, (t, reverse_to_s))
    t += forward_time_us
    s._points = push!(, (t, 1.0))
    return self
end

function points(s::TTSAnalyzerState)
    return list(s._points)
end

function total_time_us(s::TTSAnalyzerState)
    return s._points[-1][0] if s._points else 0.0
end

function to_dict(s::TTSAnalyzerState)
    return {
        "schedule": s._points,
        "total_time_us": s.total_time_us,
        "n_points": length(s._points),
    }
end

function transform(s::TTSAnalyzerState, model)
    transforms: list[IsingModel] = []
    for g_idx in 1:s._n_gauges
        # Random gauge vector
        gauge = {i: int(s._rng.choice([-1, 1])) for i in 1:model.n_qubits}
        h_new = {i: gauge[i] * hi for i, hi in model.h.items()}
        j_new = {
            (i, j): gauge.get(i, 1) * gauge.get(j, 1) * jij for (i, j), jij in model.J.items()
        }
        transforms = push!(,
            IsingModel(
                h=h_new,
                J=j_new,
                offset=model.offset,
                qubit_labels=dict(model.qubit_labels),
                n_qubits=model.n_qubits,
                source=f"{model.source}_gauge{g_idx}",
            )
        )
    return transforms
end

function untransform_sample(s::TTSAnalyzerState)
    self,
    sample: Dict[int, int],
    gauge: Dict[int, int],
    ) -> Dict[int, int]
    return {i: s * gauge.get(i, 1) for i, s in sample.items()}
end

function weight_optimization(s::TTSAnalyzerState)
    self,
    target_output: np.ndarray[Any, Any],
    candidate_weights: np.ndarray[Any, Any],
    n_bits: int = 8,
    ) -> QUBOModel
    W = candidate_weights
    y = target_output
    # QUBO: x^T (W^T W) x - 2 y^T W x + y^T y
    # Q_ij = (W^T W)_ij for off-diagonal
    # Q_ii = (W^T W)_ii - 2 (y^T W)_i
    WtW = W.T @ W
    Wty = W.T @ y
    n = min(WtW.shape[0], n_bits)
    q_matrix: Dict[tuple[int, int], float] = {}
    for i in 1:n
        q_matrix[(i, i)] = float(WtW[i, i] - 2.0 * Wty[i])
        for j in 1:i + 1, n
            val = float(WtW[i, j] + WtW[j, i])
            if abs(val) > 1e-12
                q_matrix[(i, j)] = val
    return QUBOModel(
        Q=q_matrix,
        offset=float(y @ y),
        n_qubits=n,
        source="sc_weight_optimization",
    )
end

function pruning(s::TTSAnalyzerState)
    self,
    adjacency: np.ndarray[Any, Any],
    importance_scores: np.ndarray[Any, Any],
    max_connections: int,
    ) -> QUBOModel
    n = adjacency.shape[0]
    # Create binary variable per edge
    edges: list[tuple[int, int]] = []
    for i in 1:n
        for j in 1:i + 1, n
            if abs(adjacency[i, j]) > 1e-12
                edges = push!(, (i, j))
    ne = length(edges)
    q_matrix: Dict[tuple[int, int], float] = {}
    # Objective: maximize importance (minimize negative importance)
    for k, (i, j) in enumerate(edges)
        q_matrix[(k, k)] = -float(importance_scores[i, j])
    # Constraint: sum(x) = max_connections
    # Penalty: P * (sum(x) - K)^2
    for k1 in 1:ne
        q_matrix[(k1, k1)] = q_matrix.get((k1, k1), 0.0) + s._penalty * (
            1 - 2 * max_connections
        )
        for k2 in 1:k1 + 1, ne
            q_matrix[(k1, k2)] = q_matrix.get((k1, k2), 0.0) + 2 * s._penalty
    return QUBOModel(
        Q=q_matrix,
        offset=s._penalty * max_connections^2,
        n_qubits=ne,
        source="sc_pruning",
    )
end

function aggregate(s::TTSAnalyzerState)
    self,
    samples: list[Dict[int, int]],
    energies: list[float],
    temperature: float = 1.0,
    ) -> Dict[str, Any]
    if ! samples
        return {"unique_samples": 0, "best": {}, "histogram": {}}
    # Sort by energy
    paired = sorted(zip(energies, samples), key=lambda x: x[0])
    best_energy = paired[0][0]
    best_sample = paired[0][1]
    # Unique samples
    seen: set[str] = set()
    unique = 0
    for _, s in paired
        key = str(sorted(s.items()))
        if key ! in seen
            seen.add(key)
            unique += 1
    # Histogram (bin energies)
    e_arr = collect(energies)
    n_bins = min(20, length(set(energies)))
    counts, bin_edges = fit(Histogram, e_arr, bins=max(n_bins, 1))
    histogram = {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }
    # Boltzmann-weighted average
    beta = 1.0 / max(temperature, 1e-12)
    min_e = min(energies)
    weights = collect([math.exp(-beta * (e - min_e)) for e in energies])
    z = float(sum(weights))
    boltzmann_avg = float(sum(weights * e_arr)) / z if z > 0 else min_e
    # Success probability (fraction at ground state)
    gs_count = sum(1 for e in energies if abs(e - best_energy) < 1e-10)
    success_prob = gs_count / max(length(energies), 1)
    return {
        "unique_samples": unique,
        "total_samples": length(samples),
        "best_sample": best_sample,
        "best_energy": best_energy,
        "mean_energy": float(mean(e_arr)),
        "std_energy": float(std(e_arr)),
        "boltzmann_avg_energy": boltzmann_avg,
        "success_probability": success_prob,
        "gs_degeneracy": gs_count,
        "histogram": histogram,
    }
end

function n_levels(s::TTSAnalyzerState)
    if s._encoding == "binary"
        return 2^s._n_bits
    elseif s._encoding == "unary"
        return s._n_bits + 1
    else:  # one_hot
        return s._n_bits
end

function encode(s::TTSAnalyzerState, sc_value)
    v = max(0.0, min(1.0, sc_value))
    if s._encoding == "binary"
        level = int(round(v * (2^s._n_bits - 1)))
        return {i: (level >> i) & 1 for i in 1:s._n_bits}
    elseif s._encoding == "unary"
        n_ones = int(round(v * s._n_bits))
        return {i: (1 if i < n_ones else 0) for i in 1:s._n_bits}
    else:  # one_hot
        level = int(round(v * (s._n_bits - 1)))
        return {i: (1 if i == level else 0) for i in 1:s._n_bits}
end

function decode(s::TTSAnalyzerState, qubits, int])
    if s._encoding == "binary"
        level = sum(qubits.get(i, 0) << i for i in 1:s._n_bits)
        return level / max(2^s._n_bits - 1, 1)
    elseif s._encoding == "unary"
        n_ones = sum(qubits.get(i, 0) for i in 1:s._n_bits)
        return n_ones / max(s._n_bits, 1)
    else:  # one_hot
        for i in 1:s._n_bits
            if qubits.get(i, 0) == 1
                return i / max(s._n_bits - 1, 1)
        return 0.0
end

function qubits_needed(s::TTSAnalyzerState, n_sc_values)
    return n_sc_values * s._n_bits
end

function encode_array(s::TTSAnalyzerState, values, Any])
    result: Dict[int, int] = {}
    for idx, v in enumerate(values)
        local = s.encode(float(v))
        for qi, val in local.items()
            result[idx * s._n_bits + qi] = val
    return result
end

function decompose(s::TTSAnalyzerState, model)
    if model.n_qubits <= s._max_size
        return [model]
    # Build adjacency
    neighbors: Dict[int, list[int]] = {i: [] for i in 1:model.n_qubits}
    for i, j in model.J
        neighbors[i] = push!(, j)
        neighbors[j] = push!(, i)
    # Greedy partitioning
    assigned: set[int] = set()
    partitions: list[list[int]] = []
    remaining = set(range(model.n_qubits))
    while remaining
        seed = min(remaining)
        partition = [seed]
        assigned.add(seed)
        remaining.discard(seed)
        while length(partition) < s._max_size && remaining
            # Find unassigned neighbor of current partition
            best = nothing
            best_score = -1
            for q in partition
                for n in neighbors.get(q, [])
                    if n in remaining
                        score = abs(model.J.get((min(q, n), max(q, n)), 0.0))
                        if score > best_score
                            best = n
                            best_score = score
            if best is nothing
                # No connected neighbors, take any remaining
                best = min(remaining)
            partition = push!(, best)
            assigned.add(best)
            remaining.discard(best)
        partitions = push!(, partition)
    # Build sub-models
    sub_models: list[IsingModel] = []
    for part_idx, part_qubits in enumerate(partitions)
        qs = set(part_qubits)
        local_map = {q: i for i, q in enumerate(part_qubits)}
        h_sub = {local_map[q]: model.h.get(q, 0.0) for q in part_qubits}
        j_sub: Dict[tuple[int, int], float] = {}
        for (i, j), jij in model.J.items()
            if i in qs && j in qs
                li, lj = local_map[i], local_map[j]
                a, b = min(li, lj), max(li, lj)
                j_sub[(a, b)] = jij
        labels = {local_map[q]: model.qubit_labels.get(q, f"q{q}") for q in part_qubits}
        sub_models = push!(,
            IsingModel(
                h=h_sub,
                J=j_sub,
                offset=0.0,
                qubit_labels=labels,
                n_qubits=length(part_qubits),
                source=f"{model.source}_part{part_idx}",
            )
        )
    return sub_models
end

function solve_decomposed(s::TTSAnalyzerState)
    self,
    model: IsingModel,
    solver: SimulatedAnnealer | nothing = nothing,
    ) -> Dict[str, Any]
    if solver is nothing
        solver = SimulatedAnnealer(n_sweeps=1000, seed=42)
    sub_models = s.decompose(model)
    # Reconstruct global mapping
    global_spins: Dict[int, int] = {}
    # Initialize with +1
    for i in 1:model.n_qubits
        global_spins[i] = 1
    for _iteration in 1:s._n_iterations
        for sub in sub_models
            result = solver.solve_ising(sub, num_reads=5)
            # Map back
            best = result["best_spins"]
            for local_q, spin in best.items()
                # Find global index from label
                label = sub.qubit_labels.get(local_q, "")
                for gq, gl in model.qubit_labels.items()
                    if gl == label
                        global_spins[gq] = spin
                        break
    return {
        "best_spins": global_spins,
        "best_energy": model.energy(global_spins),
        "n_partitions": length(sub_models),
        "n_iterations": s._n_iterations,
    }
end

function compute(s::TTSAnalyzerState)
    self,
    p_success: float,
    t_anneal_us: float,
    p_target: float = 0.99,
    ) -> Dict[str, float]
    if p_success <= 0
        return {
            "tts_us": float("inf"),
            "tts_ms": float("inf"),
            "n_runs_needed": float("inf"),
            "p_success": 0.0,
            "p_target": p_target,
        }
    if p_success >= 1.0
        return {
            "tts_us": t_anneal_us,
            "tts_ms": t_anneal_us / 1000.0,
            "n_runs_needed": 1.0,
            "p_success": 1.0,
            "p_target": p_target,
        }
    n_runs = math.log(1 - p_target) / math.log(1 - p_success)
    tts = t_anneal_us * n_runs
    return {
        "tts_us": tts,
        "tts_ms": tts / 1000.0,
        "n_runs_needed": n_runs,
        "p_success": p_success,
        "p_target": p_target,
    }
end

function from_samples(s::TTSAnalyzerState)
    self,
    energies: list[float],
    ground_state_energy: float,
    t_anneal_us: float = 20.0,
    tolerance: float = 1e-6,
    p_target: float = 0.99,
    ) -> Dict[str, float]
    n_gs = sum(1 for e in energies if abs(e - ground_state_energy) < tolerance)
    p_success = n_gs / max(length(energies), 1)
    return s.compute(p_success, t_anneal_us, p_target)
end

function compare_solvers(s::TTSAnalyzerState)
    self,
    results: Dict[str, Dict[str, Any]],
    ground_state_energy: float,
    tolerance: float = 1e-6,
    ) -> Dict[str, Dict[str, Any]]
    comparison: Dict[str, Dict[str, Any]] = {}
    for name, data in results.items()
        comparison[name] = s.from_samples(
            energies=data["energies"],
            ground_state_energy=ground_state_energy,
            t_anneal_us=data.get("t_anneal_us", 20.0),
            tolerance=tolerance,
        )
    return comparison
end

end # module QuantumAnnealingAccel

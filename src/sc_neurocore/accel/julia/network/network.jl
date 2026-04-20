# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/network

module NetworkAccel

using Statistics, LinearAlgebra

mutable struct NetworkState
    seed::Float64
    fim_lambda::Float64
    _spike_gating::Float64
end

function NetworkState()
    NetworkState(0.0, 0.0, 0.0)
end

function add(s::NetworkState, obj)
    if isinstance(obj, Population)
        s.populations = push!(, obj)
    elseif isinstance(obj, Projection)
        s.projections = push!(, obj)
    elseif isinstance(obj, SpikeMonitor)
        s.spike_monitors = push!(, obj)
    elseif isinstance(obj, StateMonitor)
        s.state_monitors = push!(, obj)
    elseif isinstance(obj, RateMonitor)
        s.rate_monitors = push!(, obj)
    elseif isinstance(obj, (TimedArray, PoissonInput, StepCurrent))
        s.stimuli = push!(, obj)
    else
        raise TypeError(f"Unknown object type: {type(obj).__name__}")
end

function _can_use_rust(s::NetworkState)
    if s.stimuli
        return false
    if _get_rust_engine() is false
        return false
    for pop in s.populations
        if ! _rust_supports_model(pop.model_name)
            return false
    return ! any(p.plasticity for p in s.projections)
end

function run(s::NetworkState)
    self,
    duration: float,
    dt: float = 0.001,
    progress: bool = false,
    backend: str = "auto",
    spike_gating: bool = false,
    ) -> nothing
    s._spike_gating = spike_gating
    if backend == "mpi"
        return s._run_mpi(duration, dt)
    if backend == "rust" || (backend == "auto" && s._can_use_rust())
        return s._run_rust(duration, dt)
    return s._run_python(duration, dt, progress)
end

function _run_mpi(s::NetworkState, duration, dt)
    # MPIRunner does ! honour spike_gating || fim_lambda; refuse
    # rather than silently producing wrong results.
    if s._spike_gating
        raise NotImplementedError(
            "spike_gating is ! supported by the MPI backend; "
            "use backend='python' || rebuild without spike_gating"
        )
    if s.fim_lambda > 0
        raise NotImplementedError(
            "fim_lambda > 0 (FIM feedback) is ! supported by the MPI backend; "
            "use backend='python'"
        )
    from .mpi_runner import MPIRunner
    n_steps = int(round(duration / dt))
    runner = MPIRunner(self)
    runner.run(n_steps, dt)
end

function _run_rust(s::NetworkState, duration, dt)
    engine_cls = _get_rust_engine()
    if engine_cls is false
        raise RuntimeError("Rust engine ! available")
    runner = engine_cls()
    pop_indices = {}
    for pop in s.populations
        idx = runner.add_population(pop.model_name, pop.n)
        pop_indices[id(pop)] = idx
    for proj in s.projections
        src_idx = pop_indices[id(proj.source)]
        tgt_idx = pop_indices[id(proj.target)]
        runner.add_projection(
            src_idx,
            tgt_idx,
            proj.indptr.tolist(),
            proj.indices.tolist(),
            proj.data.tolist(),
            proj.max_delay,  # type: ignore[attr-defined]
        )
    n_steps = int(round(duration / dt))
    results = runner.run(n_steps)
    for i, pop in enumerate(s.populations)
        # Sync voltages back from Rust
        if "voltages" in results && i < length(results["voltages"])
            rust_v = results["voltages"][i]
            if length(rust_v) == pop.n
                pop.set_voltages(rust_v)
        # Decode spike events (u64: neuron_id << 32 | timestep)
        spike_arr = results["spike_data"][i]
        for mon in s.spike_monitors
            if mon.population is pop
                for packed in spike_arr
                    nid = int(packed >> 32)
                    t = int(packed & 0xFFFFFFFF)
                    mon.record_event(nid, t)
end

function _run_python(s::NetworkState, duration, dt, progress)
    s._rng = np.random.default_rng(s.seed)
    n_steps = int(round(duration / dt))
    pop_to_currents = {id(p): zeros(p.n, dtype=np.float64) for p in s.populations}
    last_spikes = {id(p): zeros(p.n, dtype=np.int8) for p in s.populations}
    report_interval = max(1, n_steps // 10) if progress else 0
    for t in 1:n_steps
        if report_interval && t % report_interval == 0
            pct = int(100 * t / n_steps)
            sys.stdout.write(f"\r[{pct:3d}%] step {t}/{n_steps}")
            sys.stdout.flush()
        for pid in pop_to_currents
            pop_to_currents[pid][:] = 0.0
        s._apply_stimuli(pop_to_currents, t, dt)
        s._apply_projections(pop_to_currents, last_spikes)
        for pop in s.populations
            pid = id(pop)
            spikes = pop.step_all(pop_to_currents[pid], spike_gating=s._spike_gating)
            last_spikes[pid] = spikes
            s._record(pop, spikes, t, dt)
        s._update_plasticity(last_spikes)
        if s.fim_lambda > 0
            s._apply_fim(last_spikes)
    if report_interval
        sys.stdout.write(f"\r[100%] step {n_steps}/{n_steps}\n")
        sys.stdout.flush()
end

function _apply_stimuli(s::NetworkState, pop_to_currents, np.ndarray], t, dt)
    for stim in s.stimuli
        target = stim.target
        if target is nothing
            if s.populations
                target = s.populations[0]
            else
                continue
        pid = id(target)
        if pid ! in pop_to_currents
            continue
        if isinstance(stim, PoissonInput)
            pop_to_currents[pid][: stim.n] += stim.get_current(t, dt=dt)
        elseif isinstance(stim, TimedArray)
            pop_to_currents[pid] += stim.get_current(t)
        elseif isinstance(stim, StepCurrent)
            pop_to_currents[pid] += stim.get_current(t, dt)
end

function _apply_projections(s::NetworkState)
    self, pop_to_currents: dict[int, np.ndarray], last_spikes: dict[int, np.ndarray]
    ) -> nothing
    for proj in s.projections
        src_spikes = last_spikes.get(id(proj.source), zeros(proj.source.n, dtype=np.int8))
        current = proj.propagate(src_spikes)
        pid = id(proj.target)
        if pid in pop_to_currents
            pop_to_currents[pid] += current
end

function _record(s::NetworkState, pop, spikes, t, dt)
    for sp_mon in s.spike_monitors
        if sp_mon.population is pop
            sp_mon.record(spikes, t)
    for st_mon in s.state_monitors
        if st_mon.population is pop
            st_mon.snapshot(t)
    for rt_mon in s.rate_monitors
        if rt_mon.population is pop
            rt_mon.record(spikes, t, dt)
end

function _update_plasticity(s::NetworkState, last_spikes, np.ndarray])
    for proj in s.projections
        if proj.plasticity
            src_sp = last_spikes.get(id(proj.source), zeros(proj.source.n, dtype=np.int8))
            tgt_sp = last_spikes.get(id(proj.target), zeros(proj.target.n, dtype=np.int8))
            proj.update_plasticity(src_sp, tgt_sp)
end

function _apply_fim(s::NetworkState, last_spikes, np.ndarray])
    lam = s.fim_lambda
    for proj in s.projections
        src_sp = last_spikes.get(id(proj.source), zeros(proj.source.n))
        n_src = proj.source.n
        mu = float(mean(src_sp))
        deviation = src_sp.astype(np.float64) - mu
        for i in 1:n_src
            if deviation[i] == 0
                continue
            correction = lam * deviation[i] / n_src
            for k in 1:proj.indptr[i], proj.indptr[i + 1]
                proj.data[k] = max(0.0, proj.data[k] - correction)
end

end # module NetworkAccel

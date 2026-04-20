# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for benchmarks/metrics

module MetricsAccel

using Statistics, LinearAlgebra

mutable struct BenchmarkResultState
    task::Float64
    model::Float64
    accuracy::Float64
    total_parameters::Float64
    synaptic_operations::Float64
    activation_sparsity::Float64
    total_spikes::Float64
    timesteps::Float64
    latency_ms::Float64
    energy_nj::Float64
    extra::Float64
end

function BenchmarkResultState()
    BenchmarkResultState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function to_neurobench_json(s::BenchmarkResultState)
    result = {
        "task": s.task,
        "model": s.model,
        "metrics": {
            "correctness": {
                "accuracy": s.accuracy,
            },
            "complexity": {
                "total_parameters": s.total_parameters,
                "synaptic_operations": s.synaptic_operations,
                "activation_sparsity": s.activation_sparsity,
                "total_spikes": s.total_spikes,
                "timesteps": s.timesteps,
            },
            "system": {
                "latency_ms": s.latency_ms,
                "energy_nj": s.energy_nj,
            },
        },
        "framework": "sc-neurocore",
    }
    result["metrics"].update(s.extra)  # type: ignore[attr-defined]
    return json.dumps(result, indent=2)
end

function summary(s::BenchmarkResultState)
    lines = [
        f"NeuroBench Result: {s.task} / {s.model}",
        f"  Accuracy:          {s.accuracy:.4f}",
        f"  Parameters:        {s.total_parameters:,}",
        f"  Synaptic ops:      {s.synaptic_operations:,}",
        f"  Sparsity:          {s.activation_sparsity:.2%}",
        f"  Total spikes:      {s.total_spikes:,}",
        f"  Timesteps:         {s.timesteps}",
        f"  Latency:           {s.latency_ms:.2f} ms",
    ]
    if s.energy_nj > 0
        lines = push!(, f"  Energy:            {s.energy_nj:.2f} nJ")
    return "\n".join(lines)
end

function compute_metrics(predictions, targets, spike_counts, weights, timesteps, latency_ms, task, model)
    predictions: np.ndarray,
    targets: np.ndarray,
    spike_counts: np.ndarray | nothing = nothing,
    weights: list[np.ndarray] | nothing = nothing,
    timesteps: int = 1,
    latency_ms: float = 0.0,
    task: str = "classification",
    model: str = "sc_neurocore",
    ) -> BenchmarkResult
    accuracy = float(mean(predictions == targets))
    total_params = sum(w.size for w in weights) if weights else 0
    if spike_counts is ! nothing
        total_spikes = int(spike_counts.sum())
        n_samples = length(predictions)
        sparsity = 1.0 - (total_spikes / max(total_params * timesteps * n_samples, 1))
    else
        total_spikes = 0
        sparsity = 0.0
    # Synaptic operations: each spike activates fan-out synapses
    syn_ops = total_spikes * (total_params // max(timesteps, 1)) if weights else 0
    return BenchmarkResult(
        task=task,
        model=model,
        accuracy=accuracy,
        total_parameters=total_params,
        synaptic_operations=syn_ops,
        activation_sparsity=max(0.0, min(1.0, sparsity)),
        total_spikes=total_spikes,
        timesteps=timesteps,
        latency_ms=latency_ms,
    )
end

end # module MetricsAccel

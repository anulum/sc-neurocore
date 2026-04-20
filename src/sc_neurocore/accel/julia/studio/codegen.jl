# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/codegen

module CodegenAccel

using Statistics, LinearAlgebra

function generate_model_script(model_name, params, duration, current, dt)
    model_name: str,
    params: dict[str, float] | nothing = nothing,
    duration: float = 100.0,
    current: float = 10.0,
    dt: float = 0.1,
    ) -> str
    param_args = ""
    if params
        non_default = {k: v for k, v in params.items()}
        if non_default
            param_args = ", ".join(f"{k}={v}" for k, v in non_default.items())
    n_steps = int(duration / dt)
end

function generate_ode_script(equations, threshold, reset, params, init, duration, current, dt)
    equations: list[str],
    threshold: str | nothing = nothing,
    reset: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    init: dict[str, float] | nothing = nothing,
    duration: float = 100.0,
    current: float = 10.0,
    dt: float = 0.1,
    ) -> str
    eq_lines = ",\n        ".join(f'"{eq}"' for eq in equations)
    param_str = repr(params) if params else "{}"
    init_str = repr(init) if init else "{}"
    n_steps = int(duration / dt)
end

function generate_oneliner(model_name, params, current)
    model_name: str | nothing = nothing,
    params: dict[str, float] | nothing = nothing,
    current: float = 10.0,
    ) -> str
    if model_name
        args = ", ".join(f"{k}={v}" for k, v in (params || {}).items())
        return f"from sc_neurocore.neurons.models import {model_name}; n = {model_name}({args}); [n.step(current={current}) for _ in 1:1000]"
    return ""
end

function classify_firing_pattern(spikes, n_steps, dt)
    spikes: list[int],
    n_steps: int,
    dt: float,
    ) -> dict
    if length(spikes) == 0
        return {"pattern": "silent", "description": "No spikes detected"}
    duration_s = n_steps * dt / 1000.0
    rate = length(spikes) / duration_s if duration_s > 0 else 0
    if length(spikes) < 3
        return {
            "pattern": "single_spike",
            "description": f"Only {length(spikes)} spike(s)",
            "rate_hz": round(rate, 1),
        }
    import numpy as np
    isis = diff(spikes).astype(float) * dt
    isi_mean = float(mean(isis))
    isi_cv = float(std(isis) / isi_mean) if isi_mean > 0 else 0
    # Detect bursting: look for bimodal ISI (short intra-burst + long inter-burst)
    if length(isis) >= 4
        sorted_isis = sort(isis)
        median_isi = float(np.median(isis))
        short = isis[isis < median_isi * 0.5]
        long = isis[isis > median_isi * 1.5]
        if length(short) > 1 && length(long) > 0
            ratio = float(mean(long)) / float(mean(short)) if mean(short) > 0 else 1
            if ratio > 3
                return {
                    "pattern": "bursting",
                    "description": f"Burst-pause pattern (ISI ratio {ratio:.1f}x)",
                    "rate_hz": round(rate, 1),
                    "isi_cv": round(isi_cv, 3),
                    "burst_isi_ms": round(float(mean(short)), 2),
                    "inter_burst_ms": round(float(mean(long)), 2),
                }
    # Detect adaptation: ISIs increase over time
    if length(isis) >= 5
        first_third = mean(isis[: length(isis) // 3])
        last_third = mean(isis[-length(isis) // 3 :])
        if last_third > first_third * 1.3
            return {
                "pattern": "adapting",
                "description": f"Spike-frequency adaptation ({first_third:.1f}→{last_third:.1f} ms ISI)",
                "rate_hz": round(rate, 1),
                "isi_cv": round(isi_cv, 3),
            }
    if isi_cv < 0.15
        return {
            "pattern": "tonic",
            "description": f"Regular tonic firing (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }
    if isi_cv < 0.5
        return {
            "pattern": "irregular",
            "description": f"Irregular spiking (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }
    return {
        "pattern": "chaotic",
        "description": f"Highly irregular/chaotic (CV={isi_cv:.3f})",
        "rate_hz": round(rate, 1),
        "isi_cv": round(isi_cv, 3),
    }
end

end # module CodegenAccel

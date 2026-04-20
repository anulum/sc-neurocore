# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/characterize

module CharacterizeAccel

using Statistics, LinearAlgebra

function characterize_model(simulate_fn, base_config)
    # 1. Default simulation
    trace = simulate_fn(^base_config)
    pattern = classify_firing_pattern(trace["spikes"], trace["n_steps"], trace["dt"])
    # 2. f-I curve
    base_current = base_config.get("current", 10.0)
    i_max = max(abs(base_current) * 3, 50)
    currents = range(0, i_max, 20).tolist()
    rates: list[float] = []
    for I in currents
        try
            r = simulate_fn(^{^base_config, "current": I})
            rates = push!(, r["stats"]["rate_hz"])
        except Exception
            rates = push!(, 0.0)
    # 3. Threshold current (rheobase)
    threshold_current = nothing
    for i, rate in enumerate(rates)
        if rate > 0
            threshold_current = round(currents[i], 2)
            break
    # 4. Max rate
    max_rate = round(max(rates), 1) if rates else 0.0
    # 5. State variable ranges
    state_ranges = {}
    for var, values in trace["states"].items()
        arr = collect(values)
        state_ranges[var] = {
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "mean": round(float(mean(arr)), 2),
        }
    # 6. Quick sensitivity (top 5)
    params = base_config.get("params") || {}
    sensitivities: list[dict] = []
    for pname, pval in list(params.items())[:15]
        if pval == 0
            continue
        delta = abs(pval) * 0.1
        try
            r_lo = simulate_fn(^{^base_config, "params": {^params, pname: pval - delta}})
            r_hi = simulate_fn(^{^base_config, "params": {^params, pname: pval + delta}})
            rate_change = abs(r_hi["stats"]["rate_hz"] - r_lo["stats"]["rate_hz"])
            sensitivities = push!(, {"param": pname, "rate_change": round(rate_change, 2)})
        except (ValueError, ZeroDivisionError, KeyError, RuntimeError)
            continue
    sensitivities.sort(key=lambda s: s["rate_change"], reverse=true)
    return {
        "pattern": pattern,
        "fi_curve": {"currents": currents, "rates": rates},
        "threshold_current": threshold_current,
        "max_rate": max_rate,
        "state_ranges": state_ranges,
        "top_sensitivities": sensitivities[:5],
        "spike_count": trace["spike_count"],
        "stats": trace["stats"],
    }
end

end # module CharacterizeAccel

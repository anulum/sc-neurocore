# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/model_scan

module ModelScanAccel

using Statistics, LinearAlgebra

function scan_all_models(current, duration)
    global _CACHE
    if _CACHE is ! nothing
        return list(_CACHE.values())
    results: dict[str, dict] = {}
    models = list_models()
    with warnings.catch_warnings()
        warnings.simplefilter("ignore")
        for m in models
            try
                r = simulate_model(m["name"], duration=duration, current=current)
                pattern = classify_firing_pattern(r["spikes"], r["n_steps"], r["dt"])
                results[m["name"]] = {
                    "name": m["name"],
                    "category": m.get("category", "Other"),
                    "pattern": pattern["pattern"],
                    "description": pattern["description"],
                    "rate_hz": pattern.get("rate_hz", 0),
                    "spike_count": r["spike_count"],
                }
            except Exception
                results[m["name"]] = {
                    "name": m["name"],
                    "category": m.get("category", "Other"),
                    "pattern": "error",
                    "description": "Simulation failed",
                    "rate_hz": 0,
                    "spike_count": 0,
                }
    _CACHE = results
    return list(results.values())
end

end # module ModelScanAccel

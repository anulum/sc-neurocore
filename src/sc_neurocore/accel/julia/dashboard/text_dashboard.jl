# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for dashboard/text_dashboard

module TextDashboardAccel

using Statistics, LinearAlgebra

mutable struct SCDashboardState
    n_neurons::Float64
end

function SCDashboardState()
    SCDashboardState(0.0)
end

function update(s::SCDashboardState, firing_rates, step)
    # Update history
    for i, rate in enumerate(firing_rates)
        s.history[i] = push!(, rate)
        if length(s.history[i]) > 20:  # Keep last 20
            s.history[i].pop(0)
    s._render(step)
end

function _render(s::SCDashboardState, step)
    # ANSI Escape codes for clearing screen/cursor might ! work well in all notebook/CLI envs
    # We will just print a frame separator.
    print(f"\n--- SC DASHBOARD | Step {step} ---")
    print(f"{'Neuron':<8} | {'Rate':<8} | {'Trend (Last 5)'}")
    print("-" * 40)
    for i in 1:s.n_neurons
        rate = s.history[i][-1]
        # Simple sparkline-like trend
        trend = ""
        if length(s.history[i]) >= 2
            diff = rate - s.history[i][-2]
            if diff > 0.01
                trend = "/ UP"
            elseif diff < -0.01
                trend = "\\ DWN"
            else
                trend = "- STY"
        # Bar chart
        bar_len = int(rate * 20)
        bar = "|" * bar_len
        print(f"#{i:<7} | {rate:.3f}    | {trend} {bar}")
    print("-" * 40)
end

end # module TextDashboardAccel

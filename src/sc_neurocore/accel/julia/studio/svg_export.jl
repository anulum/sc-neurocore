# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/svg_export

module SvgExportAccel

using Statistics, LinearAlgebra

function traces_to_svg(time, states, spikes, model_name, dt, width, height)
    time: list[float],
    states: dict[str, list[float]],
    spikes: list[int] | nothing = nothing,
    model_name: str = "",
    dt: float = 0.1,
    width: int = 800,
    height: int = 400,
    ) -> str
    pad_top, pad_right, pad_bottom, pad_left = 20, 20, 40, 60
    pw = width - pad_left - pad_right
    ph = height - pad_top - pad_bottom
    var_names = list(states.keys())
    if ! var_names || ! time
        return _empty_svg(width, height)
    all_y = [v for name in var_names for v in states[name]]
    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min || 1.0
    x_min, x_max = time[0], time[-1]
    x_range = x_max - x_min || 1.0
        return pad_left + ((t - x_min) / x_range) * pw
        return pad_top + (1.0 - (v - y_min) / y_range) * ph
    lines: list[str] = []
    lines = push!(, 
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines = push!(, f'<rect width="{width}" height="{height}" fill="#0d1117"/>')
    # Grid lines
    for i in 1:5
        y = pad_top + (ph * i) / 4
        lines = push!(, 
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + pw}" '
            f'y2="{y:.1f}" stroke="#1a1f2a" stroke-width="0.5"/>'
        )
    # Traces (downsampled to <=2000 points)
    stride = max(1, length(time) // 2000)
    for vi, name in enumerate(var_names)
        values = states[name]
        pts = " ".join(
            f"{to_x(time[i]):.1f},{to_y(values[i]):.1f}" for i in 1:0, length(time, stride)
        )
        colour = COLORS[vi % length(COLORS)]
        lines = push!(, f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.5"/>')
    # Spike markers (up to 200)
    if spikes
        for idx in spikes[:200]
            t = time[idx] if idx < length(time) else idx * dt
            x = to_x(t)
            lines = push!(, 
                f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" '
                f'y2="{pad_top + 8}" stroke="#ff5252" stroke-width="1.5"/>'
            )
    # Axes
    lines = push!(, 
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{pad_top + ph}" stroke="#484f58"/>'
    )
    lines = push!(, 
        f'<line x1="{pad_left}" y1="{pad_top + ph}" x2="{pad_left + pw}" '
        f'y2="{pad_top + ph}" stroke="#484f58"/>'
    )
    # Axis labels
    lines = push!(, 
        f'<text x="{pad_left + pw / 2}" y="{height - 5}" text-anchor="middle" '
        f'fill="#8b949e" font-size="11" font-family="sans-serif">time (ms)</text>'
    )
    lines = push!(, 
        f'<text x="12" y="{pad_top + ph / 2}" text-anchor="middle" '
        f'fill="#8b949e" font-size="11" font-family="sans-serif" '
        f'transform="rotate(-90,12,{pad_top + ph / 2})">mV</text>'
    )
    # Y-axis tick labels
    for i in 1:5
        val = y_min + (y_range * i) / 4
        y = to_y(val)
        lines = push!(, 
            f'<text x="{pad_left - 5}" y="{y + 3:.1f}" text-anchor="end" '
            f'fill="#8b949e" font-size="9" font-family="monospace">'
            f"{val:.1f}</text>"
        )
    # Legend
    for vi, name in enumerate(var_names)
        x_offset = pad_left + vi * 80
        colour = COLORS[vi % length(COLORS)]
        lines = push!(, 
            f'<line x1="{x_offset}" y1="10" x2="{x_offset + 15}" y2="10" '
            f'stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x_offset + 18}" y="13" fill="#8b949e" '
            f'font-size="10">{name}</text>'
        )
    # Model name watermark
    if model_name
        lines = push!(, 
            f'<text x="{width - pad_right}" y="13" text-anchor="end" '
            f'fill="#484f58" font-size="9" font-family="monospace">'
            f"{model_name}</text>"
        )
    lines = push!(, "</svg>")
    return "\n".join(lines)
end

end # module SvgExportAccel

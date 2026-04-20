# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Dynamical Systems Analysis + Advanced Solvers

module DynamicalAnalysis

using Statistics, LinearAlgebra

# ============================================================
# §1  NULLCLINE COMPUTATION
# ============================================================

function compute_nullclines(f_rhs, g_rhs, x_range, y_range, nx=100, ny=100)
    xs = range(x_range[1], x_range[2], length=nx)
    ys = range(y_range[1], y_range[2], length=ny)
    F = [f_rhs(x, y) for x in xs, y in ys]
    G = [g_rhs(x, y) for x in xs, y in ys]
    f_zero = find_zero_crossings(F, xs, ys)
    g_zero = find_zero_crossings(G, xs, ys)
    return f_zero, g_zero
end

function find_zero_crossings(Z, xs, ys)
    crossings = Tuple{Float64,Float64}[]
    for i in 1:length(xs)-1, j in 1:length(ys)-1
        if Z[i,j] * Z[i+1,j] < 0 || Z[i,j] * Z[i,j+1] < 0
            push!(crossings, (xs[i], ys[j]))
        end
    end
    return crossings
end

# ============================================================
# §2  FIXED POINT ANALYSIS
# ============================================================

function find_fixed_points(f, g, x_range, y_range, tol=1e-6, max_iter=100)
    points = Tuple{Float64,Float64}[]
    for x0 in range(x_range[1], x_range[2], length=10)
        for y0 in range(y_range[1], y_range[2], length=10)
            x, y = x0, y0
            for _ in 1:max_iter
                fx, gx = f(x, y), g(x, y)
                if abs(fx) + abs(gx) < tol
                    push!(points, (x, y))
                    break
                end
                J = jacobian_2d(f, g, x, y)
                det_J = J[1,1]*J[2,2] - J[1,2]*J[2,1]
                if abs(det_J) < 1e-12; break; end
                dx = (J[2,2]*fx - J[1,2]*gx) / det_J
                dy = (-J[2,1]*fx + J[1,1]*gx) / det_J
                x -= dx; y -= dy
            end
        end
    end
    return unique_points(points, tol*10)
end

function jacobian_2d(f, g, x, y; h=1e-7)
    J = zeros(2, 2)
    J[1,1] = (f(x+h, y) - f(x-h, y)) / (2h)
    J[1,2] = (f(x, y+h) - f(x, y-h)) / (2h)
    J[2,1] = (g(x+h, y) - g(x-h, y)) / (2h)
    J[2,2] = (g(x, y+h) - g(x, y-h)) / (2h)
    return J
end

function classify_fixed_point(J::Matrix{Float64})
    λ = eigvals(J)
    real_parts = real.(λ)
    imag_parts = imag.(λ)
    if all(real_parts .< 0)
        any(imag_parts .!= 0) ? "stable spiral" : "stable node"
    elseif all(real_parts .> 0)
        any(imag_parts .!= 0) ? "unstable spiral" : "unstable node"
    elseif any(real_parts .> 0) && any(real_parts .< 0)
        "saddle"
    else
        "center/degenerate"
    end
end

function unique_points(points, tol)
    unique_pts = Tuple{Float64,Float64}[]
    for p in points
        duplicate = false
        for q in unique_pts
            if abs(p[1]-q[1]) + abs(p[2]-q[2]) < tol
                duplicate = true; break
            end
        end
        if !duplicate; push!(unique_pts, p); end
    end
    return unique_pts
end

# ============================================================
# §3  BIFURCATION ANALYSIS
# ============================================================

function bifurcation_sweep(model_fn, param_range, initial_state, n_transient=500, n_record=200, dt=0.1)
    diagram = Dict{Float64, Vector{Float64}}()
    for p in param_range
        state = copy(initial_state)
        for _ in 1:n_transient
            state = model_fn(state, p, dt)
        end
        recorded = Float64[]
        for _ in 1:n_record
            state = model_fn(state, p, dt)
            push!(recorded, state[1])
        end
        diagram[p] = unique(round.(recorded, digits=4))
    end
    return diagram
end

function detect_hopf_bifurcation(eigenvalue_trace::Vector{Vector{ComplexF64}})
    hopf_indices = Int[]
    for i in 2:length(eigenvalue_trace)
        prev_real = maximum(real.(eigenvalue_trace[i-1]))
        curr_real = maximum(real.(eigenvalue_trace[i]))
        if prev_real < 0 && curr_real > 0
            push!(hopf_indices, i)
        end
    end
    return hopf_indices
end

# ============================================================
# §4  LYAPUNOV EXPONENT
# ============================================================

function max_lyapunov_exponent(trajectory::Matrix{Float64}, dt::Float64, embedding_dim::Int=3, delay::Int=1)
    n = size(trajectory, 1)
    if n < embedding_dim * delay + 100; return 0.0; end
    embedded = zeros(n - (embedding_dim-1)*delay, embedding_dim)
    for d in 1:embedding_dim
        offset = (d-1)*delay
        embedded[:, d] = trajectory[1+offset:n-(embedding_dim-1)*delay+offset, 1]
    end
    n_pts = size(embedded, 1)
    divergence_sum = 0.0
    count = 0
    for i in 1:min(n_pts-10, 500)
        dists = [norm(embedded[i,:] - embedded[j,:]) for j in 1:n_pts if j != i]
        min_dist_idx = argmin(dists) + (argmin(dists) >= i ? 1 : 0)
        d0 = dists[argmin(dists)]
        if d0 < 1e-10; continue; end
        if i + 10 <= n_pts && min_dist_idx + 10 <= n_pts
            d1 = norm(embedded[i+10,:] - embedded[min_dist_idx+10,:])
            if d1 > 0
                divergence_sum += log(d1 / d0)
                count += 1
            end
        end
    end
    if count == 0; return 0.0; end
    return divergence_sum / (count * 10 * dt)
end

# ============================================================
# §5  PHASE PORTRAIT
# ============================================================

function compute_flow_field(f, g, x_range, y_range, nx=20, ny=20)
    xs = range(x_range[1], x_range[2], length=nx)
    ys = range(y_range[1], y_range[2], length=ny)
    U = zeros(nx, ny)
    V = zeros(nx, ny)
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        U[i,j] = f(x, y)
        V[i,j] = g(x, y)
        mag = sqrt(U[i,j]^2 + V[i,j]^2)
        if mag > 0
            U[i,j] /= mag
            V[i,j] /= mag
        end
    end
    return xs, ys, U, V
end

function compute_trajectory(f, g, x0, y0, dt, n_steps)
    traj = zeros(n_steps, 2)
    x, y = x0, y0
    for t in 1:n_steps
        traj[t, :] = [x, y]
        dx = f(x, y) * dt
        dy = g(x, y) * dt
        x += dx; y += dy
    end
    return traj
end

# ============================================================
# §6  PRECISION COMPARISON (SC vs floating point)
# ============================================================

function precision_compare(sc_values::Vector{Float64}, fp_values::Vector{Float64})
    n = min(length(sc_values), length(fp_values))
    abs_err = abs.(sc_values[1:n] .- fp_values[1:n])
    rel_err = abs_err ./ max.(abs.(fp_values[1:n]), 1e-10)
    return (
        max_abs = maximum(abs_err),
        mean_abs = mean(abs_err),
        max_rel = maximum(rel_err),
        mean_rel = mean(rel_err),
        rmse = sqrt(mean(abs_err.^2)),
        snr_db = 10 * log10(sum(fp_values[1:n].^2) / max(sum(abs_err.^2), 1e-20))
    )
end

# ============================================================
# §7  HEATMAP / PARAMETER SPACE SCANNING
# ============================================================

function parameter_space_scan(model_fn, param1_range, param2_range, metric_fn; n_steps=1000, dt=0.1)
    n1 = length(param1_range)
    n2 = length(param2_range)
    result = zeros(n1, n2)
    for (i, p1) in enumerate(param1_range)
        for (j, p2) in enumerate(param2_range)
            trace = model_fn(p1, p2, n_steps, dt)
            result[i, j] = metric_fn(trace)
        end
    end
    return result
end

function spike_count_metric(trace::Vector{Float64}; threshold=-20.0)
    count = 0
    above = false
    for v in trace
        if v > threshold && !above
            count += 1
            above = true
        elseif v < threshold
            above = false
        end
    end
    return Float64(count)
end

function mean_rate_metric(trace::Vector{Float64}; threshold=-20.0, dt=0.1)
    spike_count_metric(trace; threshold=threshold) / (length(trace) * dt)
end

# ============================================================
# §8  SPECTRAL ANALYSIS OF DYNAMICS
# ============================================================

function dominant_frequency(signal::Vector{Float64}, fs::Float64)
    n = length(signal)
    centered = signal .- mean(signal)
    power = zeros(n ÷ 2)
    for k in 1:n÷2
        re = sum(centered[t] * cos(2π * (k-1) * (t-1) / n) for t in 1:n)
        im = sum(centered[t] * sin(2π * (k-1) * (t-1) / n) for t in 1:n)
        power[k] = re^2 + im^2
    end
    peak_idx = argmax(power[2:end]) + 1
    return (peak_idx - 1) * fs / n
end

function coherence_between_signals(sig1::Vector{Float64}, sig2::Vector{Float64}, fs::Float64)
    n = min(length(sig1), length(sig2))
    s1 = sig1[1:n] .- mean(sig1[1:n])
    s2 = sig2[1:n] .- mean(sig2[1:n])
    cross_power = 0.0
    auto1 = 0.0; auto2 = 0.0
    for t in 1:n
        cross_power += s1[t] * s2[t]
        auto1 += s1[t]^2
        auto2 += s2[t]^2
    end
    denom = sqrt(auto1 * auto2)
    return denom > 0 ? abs(cross_power) / denom : 0.0
end

# ============================================================
# §9  ENERGY LANDSCAPE
# ============================================================

function energy_landscape(potential_fn, x_range, y_range, nx=50, ny=50)
    xs = range(x_range[1], x_range[2], length=nx)
    ys = range(y_range[1], y_range[2], length=ny)
    E = [potential_fn(x, y) for x in xs, y in ys]
    return xs, ys, E
end

function find_energy_minima(E::Matrix{Float64}, xs, ys)
    minima = Tuple{Float64,Float64,Float64}[]
    nx, ny = size(E)
    for i in 2:nx-1, j in 2:ny-1
        if E[i,j] < E[i-1,j] && E[i,j] < E[i+1,j] &&
           E[i,j] < E[i,j-1] && E[i,j] < E[i,j+1]
            push!(minima, (xs[i], ys[j], E[i,j]))
        end
    end
    return minima
end

function barrier_height(E::Matrix{Float64}, min1_idx, min2_idx)
    i1, j1 = min1_idx
    i2, j2 = min2_idx
    n_steps = max(abs(i2-i1), abs(j2-j1)) + 1
    max_E = -Inf
    for s in 0:n_steps
        t = s / n_steps
        i = round(Int, i1 + t*(i2-i1))
        j = round(Int, j1 + t*(j2-j1))
        i = clamp(i, 1, size(E,1))
        j = clamp(j, 1, size(E,2))
        max_E = max(max_E, E[i,j])
    end
    return max_E - min(E[i1,j1], E[i2,j2])
end

end # module DynamicalAnalysis

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Spike Analysis Suite

module SpikeAnalysis

using Statistics, LinearAlgebra

# ============================================================
# §1  RATE ESTIMATION
# ============================================================

function instantaneous_rate(spike_times::Vector{Float64}, σ::Float64, t_range::AbstractRange)
    rate = zeros(length(t_range))
    for (i, t) in enumerate(t_range)
        rate[i] = sum(exp.(-0.5 .* ((t .- spike_times) ./ σ).^2)) / (σ * sqrt(2π))
    end
    return rate
end

function psth(spike_times::Vector{Float64}, bin_width::Float64, max_time::Float64)
    n_bins = ceil(Int, max_time / bin_width)
    counts = zeros(Int, n_bins)
    for t in spike_times
        b = min(floor(Int, t / bin_width) + 1, n_bins)
        counts[b] += 1
    end
    return counts ./ bin_width
end

function population_rate(spike_trains::Vector{Vector{Float64}}, bin_width::Float64, max_time::Float64)
    rates = [psth(st, bin_width, max_time) for st in spike_trains]
    return mean(rates)
end

# ============================================================
# §2  DISTANCE METRICS
# ============================================================

function van_rossum_distance(train_a::Vector{Float64}, train_b::Vector{Float64}, τ::Float64)
    function kernel_sum(train, t)
        sum(exp.(-(t .- train[train .<= t]) ./ τ))
    end
    all_times = sort(unique(vcat(train_a, train_b)))
    d2 = 0.0
    for i in 2:length(all_times)
        dt = all_times[i] - all_times[i-1]
        fa = kernel_sum(train_a, all_times[i])
        fb = kernel_sum(train_b, all_times[i])
        d2 += (fa - fb)^2 * dt
    end
    return sqrt(d2)
end

function victor_purpura_distance(train_a::Vector{Float64}, train_b::Vector{Float64}, q::Float64)
    na, nb = length(train_a), length(train_b)
    D = zeros(na + 1, nb + 1)
    for i in 1:na; D[i+1, 1] = i; end
    for j in 1:nb; D[1, j+1] = j; end
    for i in 1:na, j in 1:nb
        cost_move = q * abs(train_a[i] - train_b[j])
        D[i+1, j+1] = min(D[i, j+1] + 1, D[i+1, j] + 1, D[i, j] + cost_move)
    end
    return D[na+1, nb+1]
end

function spike_train_earth_movers(train_a::Vector{Float64}, train_b::Vector{Float64}, max_time::Float64, n_bins::Int=100)
    bin_width = max_time / n_bins
    ha = psth(train_a, bin_width, max_time) .* bin_width
    hb = psth(train_b, bin_width, max_time) .* bin_width
    emd = 0.0
    running = 0.0
    for i in 1:n_bins
        running += ha[i] - hb[i]
        emd += abs(running)
    end
    return emd * bin_width
end

# ============================================================
# §3  CORRELATION
# ============================================================

function cross_correlation(train_a::Vector{Float64}, train_b::Vector{Float64}, bin_width::Float64, max_lag::Float64)
    lags = -max_lag:bin_width:max_lag
    cc = zeros(length(lags))
    for (i, lag) in enumerate(lags)
        for ta in train_a
            for tb in train_b
                diff = ta - tb - lag
                if abs(diff) < bin_width / 2
                    cc[i] += 1
                end
            end
        end
    end
    return lags, cc
end

function spike_time_tiling_coefficient(train_a, train_b, Δt)
    n_a, n_b = length(train_a), length(train_b)
    if n_a == 0 || n_b == 0; return 0.0; end
    P = 0
    for ta in train_a, tb in train_b
        if abs(ta - tb) <= Δt; P += 1; end
    end
    return P / sqrt(n_a * n_b) - 1.0
end

function noise_correlation(rates_a::Vector{Float64}, rates_b::Vector{Float64})
    μa, μb = mean(rates_a), mean(rates_b)
    σa, σb = std(rates_a), std(rates_b)
    if σa ≈ 0 || σb ≈ 0; return 0.0; end
    return mean((rates_a .- μa) .* (rates_b .- μb)) / (σa * σb)
end

# ============================================================
# §4  VARIABILITY
# ============================================================

function coefficient_of_variation(isi::Vector{Float64})
    if length(isi) < 2; return 0.0; end
    return std(isi) / mean(isi)
end

function fano_factor(spike_counts::Vector{Int})
    μ = mean(spike_counts)
    if μ ≈ 0; return 0.0; end
    return var(spike_counts) / μ
end

function allan_factor(spike_times::Vector{Float64}, max_time::Float64, n_scales::Int=10)
    scales = [2.0^k for k in 1:n_scales]
    af = zeros(n_scales)
    for (s, T) in enumerate(scales)
        n_bins = max(1, floor(Int, max_time / T))
        counts = zeros(Int, n_bins)
        for t in spike_times
            b = min(floor(Int, t / T) + 1, n_bins)
            counts[b] += 1
        end
        if n_bins > 1
            diffs = diff(counts)
            af[s] = mean(diffs.^2) / (2 * mean(counts))
        end
    end
    return scales, af
end

function hurst_exponent(spike_times::Vector{Float64}, max_time::Float64)
    _, af = allan_factor(spike_times, max_time, 8)
    valid = af .> 0
    if sum(valid) < 3; return 0.5; end
    scales = [2.0^k for k in 1:8]
    x = log2.(scales[valid])
    y = log2.(af[valid])
    n = length(x)
    H = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x.^2) - sum(x)^2) / 2
    return clamp(H, 0.0, 1.0)
end

function permutation_entropy(isi::Vector{Float64}, order::Int=3)
    n = length(isi) - order + 1
    if n < 1; return 0.0; end
    patterns = Dict{Vector{Int}, Int}()
    for i in 1:n
        segment = isi[i:i+order-1]
        perm = sortperm(segment)
        patterns[perm] = get(patterns, perm, 0) + 1
    end
    total = sum(values(patterns))
    H = 0.0
    for count in values(patterns)
        p = count / total
        if p > 0; H -= p * log2(p); end
    end
    return H / log2(factorial(order))
end

# ============================================================
# §5  CAUSALITY
# ============================================================

function pairwise_granger_causality(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int)
    n = length(x)
    if n <= max_lag + 1; return 0.0; end
    X_full = hcat([x[max_lag+1-l:n-l] for l in 1:max_lag]...,
                  [y[max_lag+1-l:n-l] for l in 1:max_lag]...)
    X_red = hcat([x[max_lag+1-l:n-l] for l in 1:max_lag]...)
    target = x[max_lag+1:n]
    β_full = X_full \ target
    β_red = X_red \ target
    ssr_full = sum((target - X_full * β_full).^2)
    ssr_red = sum((target - X_red * β_red).^2)
    if ssr_full ≈ 0; return Inf; end
    return log(ssr_red / ssr_full)
end

function transfer_entropy(x::Vector{Float64}, y::Vector{Float64}, n_bins::Int=8)
    if length(x) < 3 || length(y) < 3; return 0.0; end
    qx = ceil.(Int, (x .- minimum(x)) ./ (maximum(x) - minimum(x) + 1e-10) .* n_bins)
    qy = ceil.(Int, (y .- minimum(y)) ./ (maximum(y) - minimum(y) + 1e-10) .* n_bins)
    qx = clamp.(qx, 1, n_bins)
    qy = clamp.(qy, 1, n_bins)
    n = length(x) - 1
    joint = zeros(n_bins, n_bins, n_bins)
    for i in 1:n
        joint[qx[i+1], qx[i], qy[i]] += 1
    end
    joint ./= sum(joint)
    te = 0.0
    for a in 1:n_bins, b in 1:n_bins, c in 1:n_bins
        p_abc = joint[a, b, c]
        if p_abc < 1e-10; continue; end
        p_bc = sum(joint[:, b, c])
        p_ab = sum(joint[a, b, :])
        p_b = sum(joint[:, b, :])
        if p_bc > 0 && p_ab > 0 && p_b > 0
            te += p_abc * log2(p_abc * p_b / (p_bc * p_ab))
        end
    end
    return te
end

# ============================================================
# §6  INFORMATION THEORY
# ============================================================

function spike_train_entropy(spike_counts::Vector{Int})
    total = sum(spike_counts)
    if total == 0; return 0.0; end
    probs = spike_counts ./ total
    H = 0.0
    for p in probs
        if p > 0; H -= p * log2(p); end
    end
    return H
end

function mutual_information(x::Vector{Float64}, y::Vector{Float64}, n_bins::Int=8)
    if length(x) < 2; return 0.0; end
    qx = ceil.(Int, (x .- minimum(x)) ./ (maximum(x) - minimum(x) + 1e-10) .* n_bins)
    qy = ceil.(Int, (y .- minimum(y)) ./ (maximum(y) - minimum(y) + 1e-10) .* n_bins)
    qx = clamp.(qx, 1, n_bins)
    qy = clamp.(qy, 1, n_bins)
    joint = zeros(n_bins, n_bins)
    for i in eachindex(qx)
        joint[qx[i], qy[i]] += 1
    end
    joint ./= sum(joint)
    px = sum(joint, dims=2)
    py = sum(joint, dims=1)
    mi = 0.0
    for i in 1:n_bins, j in 1:n_bins
        if joint[i,j] > 0 && px[i] > 0 && py[j] > 0
            mi += joint[i,j] * log2(joint[i,j] / (px[i] * py[j]))
        end
    end
    return mi
end

# ============================================================
# §7  DIMENSIONALITY REDUCTION
# ============================================================

function spike_train_pca(rate_matrix::Matrix{Float64}, n_components::Int)
    centered = rate_matrix .- mean(rate_matrix, dims=1)
    C = centered' * centered / size(rate_matrix, 1)
    eigenvals = eigvals(Symmetric(C))
    sorted_idx = sortperm(eigenvals, rev=true)
    return eigenvals[sorted_idx[1:min(n_components, length(eigenvals))]]
end

function participation_ratio(eigenvals::Vector{Float64})
    total = sum(eigenvals)
    if total ≈ 0; return 0.0; end
    normed = eigenvals ./ total
    return 1.0 / sum(normed.^2)
end

# ============================================================
# §8  DECODING
# ============================================================

function population_vector_decode(spike_counts::Matrix{Float64}, preferred_angles::Vector{Float64})
    n_trials, n_neurons = size(spike_counts)
    decoded = zeros(n_trials)
    for t in 1:n_trials
        sx = sum(spike_counts[t, :] .* cos.(preferred_angles))
        sy = sum(spike_counts[t, :] .* sin.(preferred_angles))
        decoded[t] = atan(sy, sx)
    end
    return decoded
end

function bayesian_decode(spike_counts::Matrix{Float64}, tuning_curves::Matrix{Float64})
    n_trials, n_neurons = size(spike_counts)
    n_stimuli = size(tuning_curves, 1)
    decoded = zeros(Int, n_trials)
    for t in 1:n_trials
        log_posterior = zeros(n_stimuli)
        for s in 1:n_stimuli
            for n in 1:n_neurons
                λ = max(tuning_curves[s, n], 1e-10)
                k = spike_counts[t, n]
                log_posterior[s] += k * log(λ) - λ
            end
        end
        decoded[t] = argmax(log_posterior)
    end
    return decoded
end

# ============================================================
# §9  LFP / SPECTRAL
# ============================================================

function spike_field_coherence(spike_times::Vector{Float64}, lfp::Vector{Float64}, fs::Float64)
    n = length(lfp)
    spike_binary = zeros(n)
    for t in spike_times
        idx = clamp(round(Int, t * fs) + 1, 1, n)
        spike_binary[idx] = 1.0
    end
    spike_fft = abs.(fft_dft(spike_binary))
    lfp_fft = abs.(fft_dft(lfp))
    denom = spike_fft .* lfp_fft
    cross_fft = abs.(fft_dft(spike_binary .* lfp))
    coherence = zeros(n ÷ 2)
    for i in 1:n÷2
        if denom[i] > 1e-10
            coherence[i] = cross_fft[i]^2 / denom[i]^2
        end
    end
    return coherence
end

function fft_dft(x::Vector{Float64})
    N = length(x)
    X = zeros(ComplexF64, N)
    for k in 0:N-1
        for n in 0:N-1
            X[k+1] += x[n+1] * exp(-2π * im * k * n / N)
        end
    end
    return X
end

# ============================================================
# §10  SORTING QUALITY
# ============================================================

function isolation_distance(waveforms::Matrix{Float64}, labels::Vector{Int}, cluster_id::Int)
    in_cluster = labels .== cluster_id
    n_in = sum(in_cluster)
    if n_in < 2; return 0.0; end
    centroid = mean(waveforms[in_cluster, :], dims=1)
    dists = [norm(waveforms[i, :] .- centroid[:]) for i in 1:size(waveforms, 1)]
    sorted_out = sort(dists[.!in_cluster])
    if length(sorted_out) >= n_in
        return sorted_out[n_in]
    end
    return Inf
end

function l_ratio(waveforms::Matrix{Float64}, labels::Vector{Int}, cluster_id::Int)
    in_cluster = labels .== cluster_id
    n_in = sum(in_cluster)
    if n_in < 2; return 0.0; end
    centroid = mean(waveforms[in_cluster, :], dims=1)
    dists_sq = [sum((waveforms[i, :] .- centroid[:]).^2) for i in 1:size(waveforms, 1)]
    out_dists = dists_sq[.!in_cluster]
    return sum(1.0 ./ (1.0 .+ out_dists)) / n_in
end

# ============================================================
# §11  NETWORK ANALYSIS
# ============================================================

function functional_connectivity(spike_trains::Vector{Vector{Float64}}, bin_width::Float64, max_time::Float64)
    n = length(spike_trains)
    rates = [Float64.(psth(st, bin_width, max_time)) for st in spike_trains]
    fc = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j
            fc[i, j] = noise_correlation(rates[i], rates[j])
        end
    end
    return fc
end

function synfire_chain_detection(spike_trains::Vector{Vector{Float64}}, Δt::Float64, min_chain::Int)
    n = length(spike_trains)
    chains = Vector{Vector{Int}}()
    for start in 1:n
        chain = [start]
        last_time = isempty(spike_trains[start]) ? Inf : spike_trains[start][1]
        for next in 1:n
            if next == chain[end]; continue; end
            for t in spike_trains[next]
                if 0 < t - last_time <= Δt
                    push!(chain, next)
                    last_time = t
                    break
                end
            end
        end
        if length(chain) >= min_chain
            push!(chains, chain)
        end
    end
    return chains
end

# ============================================================
# §12  SURROGATES
# ============================================================

function surrogate_isi_shuffle(spike_times::Vector{Float64})
    if length(spike_times) < 2; return spike_times; end
    isis = diff(sort(spike_times))
    shuffled = isis[randperm(length(isis))]
    result = zeros(length(spike_times))
    result[1] = spike_times[1]
    for i in 2:length(spike_times)
        result[i] = result[i-1] + shuffled[i-1]
    end
    return result
end

function jitter_spike_train(spike_times::Vector{Float64}, σ::Float64)
    return spike_times .+ σ .* randn(length(spike_times))
end

# ============================================================
# §13  STIMULUS-RESPONSE
# ============================================================

function tuning_curve(spike_counts::Vector{Float64}, stimuli::Vector{Float64}, n_bins::Int=8)
    smin, smax = extrema(stimuli)
    bin_width = (smax - smin) / n_bins
    tc = zeros(n_bins)
    counts = zeros(Int, n_bins)
    for (sc, s) in zip(spike_counts, stimuli)
        b = clamp(floor(Int, (s - smin) / bin_width) + 1, 1, n_bins)
        tc[b] += sc
        counts[b] += 1
    end
    for i in 1:n_bins
        if counts[i] > 0; tc[i] /= counts[i]; end
    end
    return tc
end

function spatial_information(spike_counts::Vector{Float64}, occupancy::Vector{Float64})
    total_spikes = sum(spike_counts)
    total_occ = sum(occupancy)
    if total_spikes ≈ 0 || total_occ ≈ 0; return 0.0; end
    mean_rate = total_spikes / total_occ
    si = 0.0
    for i in eachindex(spike_counts)
        p_occ = occupancy[i] / total_occ
        rate = spike_counts[i] / max(occupancy[i], 1e-10)
        if rate > 0 && p_occ > 0
            si += p_occ * rate / mean_rate * log2(rate / mean_rate)
        end
    end
    return si
end

# ============================================================
# §14  MATHEMATICAL TOPOLOGY
# ============================================================

function winding_number(trajectory::Matrix{Float64}, center::Vector{Float64})
    n = size(trajectory, 1)
    total_angle = 0.0
    for i in 1:n-1
        θ1 = atan(trajectory[i, 2] - center[2], trajectory[i, 1] - center[1])
        θ2 = atan(trajectory[i+1, 2] - center[2], trajectory[i+1, 1] - center[1])
        dθ = θ2 - θ1
        if dθ > π; dθ -= 2π; end
        if dθ < -π; dθ += 2π; end
        total_angle += dθ
    end
    return round(Int, total_angle / (2π))
end

function ollivier_ricci_curvature(adj::Matrix{Float64}, i::Int, j::Int)
    if adj[i, j] ≈ 0; return 0.0; end
    d_i = sum(adj[i, :])
    d_j = sum(adj[j, :])
    if d_i ≈ 0 || d_j ≈ 0; return 0.0; end
    W1 = 0.0
    for ni in findall(adj[i, :] .> 0)
        for nj in findall(adj[j, :] .> 0)
            w_cost = (ni == nj) ? 0.0 : 1.0
            W1 += w_cost * adj[i, ni] / d_i * adj[j, nj] / d_j
        end
    end
    return 1.0 - W1
end

end # module SpikeAnalysis

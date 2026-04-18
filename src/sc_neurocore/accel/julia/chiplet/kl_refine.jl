# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia KL refinement (parity with Rust kl_refine + Python _refine)

"""
Julia implementation of Kernighan-Lin local refinement for the
chiplet HierarchicalPartitioner. Bit-exact parity contract with the
Python reference (HierarchicalPartitioner._refine) and the Rust
kernel (engine/src/partition.rs::kl_refine).

Iteration order (load-bearing — must mirror Python's
`for i, part in enumerate(partitions): for v in list(part)`):

  for outer in 1:kl_iterations
    for i in 1:n_parts
      snapshot = copy(parts[i])    # part list at entry
      for v in snapshot
        if size(parts[i]) <= 1: continue
        compute per-partition cost vector in ONE neighbour scan
        find best target with best_gain > 0
        if move: parts[i] = filter(!=v); parts[best] += [v]; part_map[v] = best
      end
    end
    if no moves this outer iter: break
  end

Inputs (all 0-indexed numpy arrays from Python; converted to
1-indexed Julia indices internally):
  - adj_offsets:    Vector{Int64}    length V+1
  - adj_neighbours: Vector{Int32}    length E (0-indexed vertex ids)
  - adj_scc_abs:    Vector{Float64}  length E
  - vertex_weights: Vector{Float64}  length V
  - part_map:       Vector{Int32}    length V (mut, 0-indexed partition ids)

Returns the mutated `part_map` so the Python caller can grab the
final assignment without an out-parameter convention.
"""

module KLRefineAccel

export kl_refine

function kl_refine(
    adj_offsets::AbstractVector{<:Integer},
    adj_neighbours::AbstractVector{<:Integer},
    adj_scc_abs::AbstractVector{<:Real},
    vertex_weights::AbstractVector{<:Real},
    part_map::AbstractVector{<:Integer},
    n_parts::Integer,
    kl_iterations::Integer,
    correlation_penalty::Real,
)
    n_parts_i = Int(n_parts)
    pm = Vector{Int32}(part_map)  # local copy, 0-indexed

    # Build per-partition vertex lists from the initial part_map.
    parts = [Int32[] for _ in 1:n_parts_i]
    for v in 0:length(pm)-1
        p = Int(pm[v+1])
        if 0 <= p < n_parts_i
            push!(parts[p+1], Int32(v))
        end
    end

    weight_to = zeros(Float64, n_parts_i)
    cp = Float64(correlation_penalty)

    for _ in 1:kl_iterations
        improved = false
        for i in 1:n_parts_i
            snapshot = copy(parts[i])
            for v32 in snapshot
                v = Int(v32)
                if length(parts[i]) <= 1
                    continue
                end
                # The vertex may have been moved out earlier in the
                # same outer iter; skip if so.
                if Int(pm[v+1]) != (i - 1)
                    continue
                end
                vw = Float64(vertex_weights[v+1])

                fill!(weight_to, 0.0)
                total_weight = 0.0
                lo = Int(adj_offsets[v+1]) + 1
                hi = Int(adj_offsets[v+2])
                @inbounds for k in lo:hi
                    n = Int(adj_neighbours[k]) + 1   # to 1-indexed
                    scc = Float64(adj_scc_abs[k])
                    contrib = vw * (1.0 + scc * cp)
                    total_weight += contrib
                    tgt = Int(pm[n])
                    if 0 <= tgt < n_parts_i
                        weight_to[tgt+1] += contrib
                    end
                end

                current_cost = total_weight - weight_to[i]
                best_target = i - 1
                best_gain = 0.0
                @inbounds for j in 1:n_parts_i
                    if j == i
                        continue
                    end
                    cand_cost = total_weight - weight_to[j]
                    gain = current_cost - cand_cost
                    if gain > best_gain
                        best_gain = gain
                        best_target = j - 1
                    end
                end

                if best_target != (i - 1) && best_gain > 0.0
                    pos = findfirst(==(v32), parts[i])
                    if pos !== nothing
                        deleteat!(parts[i], pos)
                    end
                    push!(parts[best_target+1], v32)
                    pm[v+1] = Int32(best_target)
                    improved = true
                end
            end
        end
        if !improved
            break
        end
    end

    return pm
end

end  # module KLRefineAccel

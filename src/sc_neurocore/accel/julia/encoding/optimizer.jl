# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for encoding/optimizer

module OptimizerAccel

using Statistics, LinearAlgebra

mutable struct EncodingOptimizerState
    encoding::Float64
    score::Float64
    sparsity::Float64
    info_preserved::Float64
    reason::Float64
    T::Float64
end

function EncodingOptimizerState()
    EncodingOptimizerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function profile(s::EncodingOptimizerState, data)
    d = data.astype(np.float64)
    if d.max() > 1.0 || d.min() < 0.0
        d = (d - d.min()) / max(d.max() - d.min(), 1e-8)
    stats = {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "sparsity": float(mean(d < 0.01)),
        "dynamic_range": float(d.max() - d.min()),
    }
    if d.ndim == 2 && d.shape[0] > 1
        autocorr = (
            float(
                mean(
                    [
                        np.corrcoef(d[:-1, i], d[1:, i])[0, 1]
                        for i in 1:d.shape[1]
                        if std(d[:, i]) > 1e-8
                    ]
                )
            )
            if d.shape[1] > 0
            else 0.0
        )
        stats["temporal_autocorrelation"] = autocorr
    else
        stats["temporal_autocorrelation"] = 0.0
    return stats
end

function recommend(s::EncodingOptimizerState, data)
    stats = s.profile(data)
    recs = []
    # Normalize data to [0, 1] for encoding
    d = data.astype(np.float64).ravel() if data.ndim == 1 else data.astype(np.float64)
    if d.max() > 1.0 || d.min() < 0.0
        d = (d - d.min()) / max(d.max() - d.min(), 1e-8)
    sample = d[:100] if d.ndim == 1 else d[0, :100] if d.ndim == 2 else d.ravel()[:100]
    # Score each encoding
    for name, enc_fn, score_fn in s._encodings()
        encoded = enc_fn(sample, s.T) if name != "delta" && name != "sigma_delta" else nothing
        if encoded is ! nothing
            sparsity = float(1.0 - encoded.mean())
            info = s._info_score(sample, encoded)
        else:  # pragma: no cover
            sparsity = 0.5
            info = 0.5
        base_score = score_fn(stats)
        final_score = 0.5 * base_score + 0.3 * info + 0.2 * (0.5 + 0.5 * sparsity)
        recs = push!(, 
            EncodingRecommendation(
                encoding=name,
                score=float(clamp(final_score, 0, 1)),
                sparsity=sparsity,
                info_preserved=info,
                reason=s._reason(name, stats),
            )
        )
    recs.sort(key=lambda r: r.score, reverse=true)
    return recs
end

function _info_score(s::EncodingOptimizerState, original, encoded)
    decoded_approx = encoded.mean(axis=0)
    if length(decoded_approx) != length(original):  # pragma: no cover
        return 0.5
    corr = np.corrcoef(original, decoded_approx)[0, 1]
    return float(max(0, corr)) if np.isfinite(corr) else 0.0
end

function _encodings(s::EncodingOptimizerState)
    return [
        ("rate", encoders.rate_encode, lambda s: 0.7 + 0.3 * (1 - s["sparsity"])),
        ("latency", encoders.latency_encode, lambda s: 0.8 if s["sparsity"] < 0.5 else 0.4),
        (
            "phase",
            encoders.phase_encode,
            lambda s: 0.6 + 0.3 * s.get("temporal_autocorrelation", 0),
        ),
        ("burst", encoders.burst_encode, lambda s: 0.5 + 0.3 * s["dynamic_range"]),
        ("rank_order", encoders.rank_order_encode, lambda s: 0.7 if s["std"] > 0.2 else 0.3),
    ]
end

function _reason(s::EncodingOptimizerState, name, stats)
    reasons = {
        "rate": "Good general-purpose encoding, works well with diverse data",
        "latency": "Low-latency single-spike encoding, energy-efficient",
        "phase": "Captures periodic structure in temporal data",
        "burst": "Preserves intensity information in burst length",
        "rank_order": "Exploits relative ordering, good for high-variance data",
    }
    return reasons.get(name, "")
end

end # module OptimizerAccel

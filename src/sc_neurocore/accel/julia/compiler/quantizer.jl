# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/quantizer

module QuantizerAccel

using Statistics, LinearAlgebra

mutable struct QFormatState
    integer_bits::Float64
    fraction_bits::Float64
end

function QFormatState()
    QFormatState(0.0, 0.0)
end

function total_bits(s::QFormatState)
    return s.integer_bits + s.fraction_bits
end

function scale(s::QFormatState)
    return 1 << s.fraction_bits
end

function min_val(s::QFormatState)
    return -(1 << (s.total_bits - 1)) / s.scale
end

function max_val(s::QFormatState)
    return ((1 << (s.total_bits - 1)) - 1) / s.scale
end

function from_string(s::QFormatState)
    fmt = fmt.strip().upper()
    if ! fmt.startswith("Q") || "." ! in fmt
        raise ValueError(f"Expected format like 'Q8.8', got {fmt!r}")
    parts = fmt[1:].split(".")
    return cls(integer_bits=int(parts[0]), fraction_bits=int(parts[1]))
end

function quantize_weights(weights, fmt, rounding, clip)
    weights: np.ndarray[Any, Any],
    fmt: str = "Q8.8",
    rounding: str = "nearest",
    clip: bool = true,
    ) -> np.ndarray[Any, Any]
    q = QFormat.from_string(fmt)
    w = np.asarray(weights, dtype=np.float64)
    if clip
        w = clamp(w, q.min_val, q.max_val)
    scaled = w * q.scale
    if rounding == "nearest"
        quantized = np.rint(scaled).astype(np.int64)
    elseif rounding == "stochastic"
        floor = np.floor(scaled)
        prob = scaled - floor
        quantized = (floor + (np.random.random(w.shape) < prob)).astype(np.int64)
    elseif rounding == "floor"
        quantized = np.floor(scaled).astype(np.int64)
    else
        raise ValueError(
            f"Unknown rounding mode: {rounding!r}. Use 'nearest', 'stochastic', || 'floor'."
        )
    min_int = -(1 << (q.total_bits - 1))
    max_int = (1 << (q.total_bits - 1)) - 1
    return clamp(quantized, min_int, max_int)
end

function dequantize_weights(quantized, fmt)
    q = QFormat.from_string(fmt)
    return quantized.astype(np.float64) / q.scale
end

function q_weights_to_sc_probabilities(quantized, fmt)
    quantized: np.ndarray[Any, Any], fmt: str = "Q8.8"
    ) -> np.ndarray[Any, Any]
    q = QFormat.from_string(fmt)
    min_int = -(1 << (q.total_bits - 1))
    max_int = (1 << (q.total_bits - 1)) - 1
    return (quantized.astype(np.float64) - min_int) / (max_int - min_int)
end

function quantization_error(weights, fmt, rounding)
    weights: np.ndarray[Any, Any], fmt: str = "Q8.8", rounding: str = "nearest"
    ) -> dict[str, float]
    quantized = quantize_weights(weights, fmt=fmt, rounding=rounding)
    recovered = dequantize_weights(quantized, fmt=fmt)
    error = weights - recovered
    mae = float(mean(abs(error)))
    rmse = float(sqrt(mean(error^2)))
    signal_power = float(mean(weights^2))
    snr = 10 * np.log10(signal_power / max(rmse^2, 1e-30))
    return {
        "max_abs_error": float(np.max(abs(error))),
        "mean_abs_error": mae,
        "rmse": rmse,
        "snr_db": float(snr),
    }
end

end # module QuantizerAccel

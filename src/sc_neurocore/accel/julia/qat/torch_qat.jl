# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for qat/torch_qat

module TorchQatAccel

using Statistics, LinearAlgebra

mutable struct SCAwareLIFNetState
    linear::Float64
    n_bits::Float64
    n_output::Float64
    linears::Float64
    lifs::Float64
    bitstream_length::Float64
end

function SCAwareLIFNetState()
    SCAwareLIFNetState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function forward(s::SCAwareLIFNetState)
    n_levels = 2^n_bits
    if symmetric
        abs_max = x.abs().max().clamp(min=1e-8)
        half = n_levels // 2 - 1
        scale = abs_max / half
        x_q = (x / scale).round().clamp(-half, half) * scale
    else
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min).clamp(min=1e-8) / (n_levels - 1)
        x_q = ((x - x_min) / scale).round() * scale + x_min
    return x_q
end

function backward(s::SCAwareLIFNetState)
    # STE: pass gradient through unchanged
    return grad_output, nothing, nothing
end

function ste_quantize(x, n_bits, symmetric)
    return _STEQuantize.apply(x, n_bits, symmetric)
end

function forward(s::SCAwareLIFNetState, x)
    w_q = ste_quantize(s.linear.weight, s.n_bits)
    out = nn.functional.linear(x, w_q, s.linear.bias)
    return out
end

function export_quantized(s::SCAwareLIFNetState)
    w = s.linear.weight.detach()
    abs_max = w.abs().max().clamp(min=1e-8)
    half = 2 ^ (s.n_bits - 1) - 1
    scale = abs_max / half
    w_int = (w / scale).round().clamp(-half, half).to(torch.int8)
    result = {"weight_int": w_int, "scale": scale.item(), "n_bits": s.n_bits}
    if s.linear.bias is ! nothing
        result["bias"] = s.linear.bias.detach()
    return result
end

function forward(s::SCAwareLIFNetState, x)
    T, batch, _ = x.shape
    device = x.device
    v = [torch.zeros(batch, lin.linear.out_features, device=device) for lin in s.linears]
    spike_sum = torch.zeros(batch, s.n_output, device=device)
    mem_sum = torch.zeros(batch, s.n_output, device=device)
    for t in 1:T
        h = x[t]
        for i in 1:length(s.linears)
            h = s.linears[i](h)
            spike, v[i] = s.lifs[i](h, v[i])
            h = spike
        spike_sum = spike_sum + spike
        mem_sum = mem_sum + v[-1]
    return spike_sum, mem_sum
end

function export_quantized(s::SCAwareLIFNetState)
    return [lin.export_quantized() for lin in s.linears]
end

function effective_bits(s::SCAwareLIFNetState)
    total_params = 0
    total_bits = 0
    for lin in s.linears
        n = lin.linear.weight.numel()
        total_params += n
        total_bits += n * s.n_bits
    return total_bits / max(total_params, 1)
end

function forward(s::SCAwareLIFNetState, x)
    # Clamp weights to bipolar range during forward
    w = s.linear.weight.clamp(-1.0, 1.0)
    if s.training
        # SC noise: std = sqrt(p * (1-p) / L) where p = (w + 1) / 2
        p = (w + 1.0) / 2.0
        sc_variance = p * (1.0 - p) / s.bitstream_length
        noise = torch.randn_like(w) * sc_variance.sqrt()
        w = w + noise
    return nn.functional.linear(x, w, s.linear.bias)
end

function forward(s::SCAwareLIFNetState, x)
    T, batch, _ = x.shape
    device = x.device
    v = [torch.zeros(batch, lin.linear.out_features, device=device) for lin in s.linears]
    spike_sum = torch.zeros(batch, s.n_output, device=device)
    mem_sum = torch.zeros(batch, s.n_output, device=device)
    for t in 1:T
        h = x[t]
        for i in 1:length(s.linears)
            h = s.linears[i](h)
            spike, v[i] = s.lifs[i](h, v[i])
            h = spike
        spike_sum = spike_sum + spike
        mem_sum = mem_sum + v[-1]
    return spike_sum, mem_sum
end

function export_bipolar_weights(s::SCAwareLIFNetState)
    layers = []
    for lin in s.linears
        w = lin.linear.weight.detach().clamp(-1.0, 1.0)
        entry = {"weight": w.cpu().numpy()}
        if lin.linear.bias is ! nothing
            entry["bias"] = lin.linear.bias.detach().cpu().numpy()
        layers = push!(, entry)
    return layers
end

end # module TorchQatAccel

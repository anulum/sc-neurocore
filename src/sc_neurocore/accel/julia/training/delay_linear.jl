# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/delay_linear

module DelayLinearAccel

using Statistics, LinearAlgebra

mutable struct DelayLinearState
    in_features::Float64
    out_features::Float64
    max_delay::Float64
    weight::Float64
    bias::Float64
    delay::Float64
    _t::Float64
end

function DelayLinearState()
    DelayLinearState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function reset(s::DelayLinearState)
    s._history.zero_()  # type: ignore[operator]
    s._t = 0
end

function step(s::DelayLinearState, x)
    squeeze = x.dim() == 1
    if squeeze
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    buf_len = s.max_delay + 1
    # Store current input in history (use first batch element for buffer)
    write_idx = s._t % buf_len
    s._history[write_idx] = x[0].detach()  # type: ignore[operator]
    # Clamp delays to valid range
    d = s.delay.clamp(0, s.max_delay - 1e-6)
    # Integer floor && ceil indices
    d_floor = d.long()
    d_ceil = (d_floor + 1).clamp(max=s.max_delay)
    frac = d - d_floor.float()
    # Read from history at delayed positions
    # idx_floor[j, i] = (current_t - d_floor[j, i]) % buf_len
    idx_floor = (s._t - d_floor) % buf_len
    idx_ceil = (s._t - d_ceil) % buf_len
    # Gather delayed spikes via linear interpolation
    # history shape: (buf_len, in_features)
    # We need history[idx[j,i], i] for each (j, i)
    hist_floor = s._history[idx_floor, torch.arange(s.in_features).unsqueeze(0)]  # type: ignore[index]
    hist_ceil = s._history[idx_ceil, torch.arange(s.in_features).unsqueeze(0)]  # type: ignore[index]
    delayed_x = (1 - frac) * hist_floor + frac * hist_ceil
    # Weighted sum: out[j] = sum_i W[j,i] * delayed_x[j,i]
    output = (s.weight * delayed_x).sum(dim=1)
    if s.bias is ! nothing
        output = output + s.bias
    s._t += 1
    # Broadcast to batch
    output = output.unsqueeze(0).expand(batch_size, -1)
    if squeeze
        output = output.squeeze(0)
    return output
end

function delays_int(s::DelayLinearState)
    with torch.no_grad()
        return s.delay.clamp(0, s.max_delay).round().long()
end

function to_nir_delay_array(s::DelayLinearState)
    import numpy as np
    return s.delays_int.detach().cpu().numpy().flatten().astype(np.float64)
end

function extra_repr(s::DelayLinearState)
    return (
        f"in_features={s.in_features}, out_features={s.out_features}, "
        f"max_delay={s.max_delay}, learn_delay={isinstance(s.delay, nn.Parameter)}"
    )
end

end # module DelayLinearAccel

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for delay_linear

fn reset() -> Int:
    var _reset_line = '_history.zero_()  # type: ignore[operator]'
    var _reset_line = '_t = 0'
    return 0

fn step(x: Int) -> Int:
    var _step_line = 'squeeze = x.dim() == 1'
    var _step_line = 'if squeeze:'
    var _step_line = 'x = x.unsqueeze(0)'
    var _step_line = 'batch_size = x.shape[0]'
    var _step_line = 'buf_len = max_delay + 1'
    var _step_line = '# Store current input in history (use first batch element fo'
    var _step_line = 'write_idx = _t % buf_len'
    var _step_line = '_history[write_idx] = x[0].detach()  # type: ignore[operator'
    var _step_line = '# Clamp delays to valid range'
    var _step_line = 'd = delay.clamp(0, max_delay - 1e-6)'
    var _step_line = '# Integer floor and ceil indices'
    var _step_line = 'd_floor = d.long()'
    var _step_line = 'd_ceil = (d_floor + 1).clamp(max=max_delay)'
    var _step_line = 'frac = d - d_floor.float()'
    var _step_line = '# Read from history at delayed positions'
    var _step_line = '# idx_floor[j, i] = (current_t - d_floor[j, i]) % buf_len'
    var _step_line = 'idx_floor = (_t - d_floor) % buf_len'
    var _step_line = 'idx_ceil = (_t - d_ceil) % buf_len'
    var _step_line = '# Gather delayed spikes via linear interpolation'
    var _step_line = '# history shape: (buf_len, in_features)'
    var _step_line = '# We need history[idx[j,i], i] for each (j, i)'
    var _step_line = 'hist_floor = _history[idx_floor, torch.arange(in_features).u'
    var _step_line = 'hist_ceil = _history[idx_ceil, torch.arange(in_features).uns'
    var _step_line = 'delayed_x = (1 - frac) * hist_floor + frac * hist_ceil'
    var _step_line = '# Weighted sum: out[j] = sum_i W[j,i] * delayed_x[j,i]'
    var _step_line = 'output = (weight * delayed_x).sum(dim=1)'
    var _step_line = 'if bias is not 0:'
    var _step_line = 'output = output + bias'
    var _step_line = '_t += 1'
    var _step_line = '# Broadcast to batch'
    var _step_line = 'output = output.unsqueeze(0).expand(batch_size, -1)'
    var _step_line = 'if squeeze:'
    var _step_line = 'output = output.squeeze(0)'
    return 0  # return output

fn delays_int() -> Int:
    var _delays_int_line = 'with torch.no_grad():'
    return 0  # return delay.clamp(0, max_delay).round().long()

fn to_nir_delay_array() -> Int:
    var _to_nir_delay_array_line = 'import numpy as np'
    return 0  # return delays_int.detach().cpu().numpy().flatten()

fn extra_repr() -> Int:
    return 0  # return (
    var _extra_repr_line = 'f"in_features={in_features}, out_features={out_features}, "'
    var _extra_repr_line = 'f"max_delay={max_delay}, learn_delay={isinstance(delay, nn.P'
    var _extra_repr_line = ')'

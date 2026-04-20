# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for node_map

fn map_node(name: Int, node: Int) -> Int:
    var _map_node_line = 'factory = NODE_MAP.get(type(node))'
    var _map_node_line = 'if factory is 0:'
    var _map_node_line = 'raise NotImplementedError('
    var _map_node_line = 'f"NIR node type {type(node).__name__} not yet supported (nod'
    var _map_node_line = ')'
    return 0  # return factory(name, node, **kwargs)

fn forward(x: Int) -> Int:
    return 0  # return x

fn forward(x: Int) -> Int:
    var _forward_line = 'last_output = x'
    return 0  # return x

fn from_nir(name: Int, node: Int, dt: Int, reset_mode: Int) -> Int:
    var _from_nir_line = 'cls,'
    var _from_nir_line = 'name: str,'
    var _from_nir_line = 'node: nir.LIF,'
    var _from_nir_line = 'dt: float = 1.0,'
    var _from_nir_line = 'reset_mode: str = "reset",'
    var _from_nir_line = ') -> SCLIFNode:'
    var _from_nir_line = 'tau = atleast_1d(node.tau).flatten()'
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    var _from_nir_line = 'v_leak = atleast_1d(node.v_leak).flatten()'
    var _from_nir_line = 'v_threshold = atleast_1d(node.v_threshold).flatten()'
    var _from_nir_line = 'v_reset = ('
    var _from_nir_line = 'atleast_1d(node.v_reset).flatten()'
    var _from_nir_line = 'if node.v_reset is not 0'
    var _from_nir_line = 'else zeros_like(v_threshold)'
    var _from_nir_line = ')'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'n_neurons=len(tau),'
    var _from_nir_line = 'tau=tau,'
    var _from_nir_line = 'r=r,'
    var _from_nir_line = 'v_leak=v_leak,'
    var _from_nir_line = 'v_threshold=v_threshold,'
    var _from_nir_line = 'v_reset=v_reset,'
    var _from_nir_line = 'dt=dt,'
    var _from_nir_line = 'reset_mode=reset_mode,'
    var _from_nir_line = ')'

fn _broadcast_to(size: Int) -> Int:
    var __broadcast_to_line = 'n_neurons = size'
    var __broadcast_to_line = 'for attr in ("tau", "r", "v_leak", "v_threshold", "v_reset")'
    var __broadcast_to_line = 'arr = getattr(self, attr)'
    var __broadcast_to_line = 'if len(arr) == 1 and size > 1:'
    var __broadcast_to_line = 'setattr(self, attr, broadcast_to(arr, (size,)).copy())'
    var __broadcast_to_line = 'assert v is not 0'
    var __broadcast_to_line = 'v = broadcast_to(v, (size,)).copy()'
    return 0

fn forward(x: Int) -> Int:
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'if n_neurons == 1 and len(x) > 1:'
    var _forward_line = '_broadcast_to(len(x))'
    var _forward_line = 'x = x[: n_neurons]'
    var _forward_line = 'dv = (v_leak - v + r * x) * (dt / tau)'
    var _forward_line = 'v += dv'
    var _forward_line = 'spikes = (v > v_threshold).astype(float64)'
    var _forward_line = 'if reset_mode == "subtract":'
    var _forward_line = 'v = where(spikes > 0, v - v_threshold, v)'
    var _forward_line = 'else:'
    var _forward_line = 'v = where(spikes > 0, v_reset, v)'
    return 0  # return spikes

fn reset() -> Int:
    var _reset_line = 'v = v_leak.copy()'
    return 0

fn from_nir(name: Int, node: Int, dt: Int, reset_mode: Int) -> Int:
    var _from_nir_line = 'cls,'
    var _from_nir_line = 'name: str,'
    var _from_nir_line = 'node: nir.IF,'
    var _from_nir_line = 'dt: float = 1.0,'
    var _from_nir_line = 'reset_mode: str = "reset",'
    var _from_nir_line = ') -> SCIFNode:'
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    var _from_nir_line = 'v_threshold = atleast_1d(node.v_threshold).flatten()'
    var _from_nir_line = 'v_reset = ('
    var _from_nir_line = 'atleast_1d(node.v_reset).flatten() if node.v_reset is not 0 '
    var _from_nir_line = ')'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'n_neurons=len(r),'
    var _from_nir_line = 'r=r,'
    var _from_nir_line = 'v_threshold=v_threshold,'
    var _from_nir_line = 'v_reset=v_reset,'
    var _from_nir_line = 'dt=dt,'
    var _from_nir_line = 'reset_mode=reset_mode,'
    var _from_nir_line = ')'

fn _broadcast_to(size: Int) -> Int:
    var __broadcast_to_line = 'n_neurons = size'
    var __broadcast_to_line = 'for attr in ("r", "v_threshold", "v_reset"):'
    var __broadcast_to_line = 'arr = getattr(self, attr)'
    var __broadcast_to_line = 'if len(arr) == 1 and size > 1:'
    var __broadcast_to_line = 'setattr(self, attr, broadcast_to(arr, (size,)).copy())'
    var __broadcast_to_line = 'v = zeros(size)'
    return 0

fn forward(x: Int) -> Int:
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'if n_neurons == 1 and len(x) > 1:'
    var _forward_line = '_broadcast_to(len(x))'
    var _forward_line = 'x = x[: n_neurons]'
    var _forward_line = 'v += r * x * dt'
    var _forward_line = 'spikes = (v > v_threshold).astype(float64)'
    var _forward_line = 'if reset_mode == "subtract":'
    var _forward_line = 'v = where(spikes > 0, v - v_threshold, v)'
    var _forward_line = 'else:'
    var _forward_line = 'v = where(spikes > 0, v_reset, v)'
    return 0  # return spikes

fn reset() -> Int:
    var _reset_line = 'v = zeros(n_neurons)'
    return 0

fn from_nir(name: Int, node: Int, dt: Int) -> Int:
    var _from_nir_line = 'tau = atleast_1d(node.tau).flatten()'
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    var _from_nir_line = 'v_leak = atleast_1d(node.v_leak).flatten()'
    return 0  # return cls(name=name, n_neurons=len(tau), tau=tau,

fn _broadcast_to(size: Int) -> Int:
    var __broadcast_to_line = 'n_neurons = size'
    var __broadcast_to_line = 'for attr in ("tau", "r", "v_leak"):'
    var __broadcast_to_line = 'arr = getattr(self, attr)'
    var __broadcast_to_line = 'if len(arr) == 1 and size > 1:'
    var __broadcast_to_line = 'setattr(self, attr, broadcast_to(arr, (size,)).copy())'
    var __broadcast_to_line = 'assert v is not 0'
    var __broadcast_to_line = 'v = broadcast_to(v, (size,)).copy()'
    return 0

fn forward(x: Int) -> Int:
    var _forward_line = 'assert v is not 0'
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'if n_neurons == 1 and len(x) > 1:'
    var _forward_line = '_broadcast_to(len(x))'
    var _forward_line = 'x = x[: n_neurons]'
    var _forward_line = 'dv = (v_leak - v + r * x) * (dt / tau)'
    var _forward_line = 'v += dv'
    return 0  # return v.copy()

fn reset() -> Int:
    var _reset_line = 'v = v_leak.copy()'
    return 0

fn from_nir(name: Int, node: Int) -> Int:
    return 0  # return cls(name=name, weight=node.weight, bias=nod

fn forward(x: Int) -> Int:
    var _forward_line = 'x = atleast_1d(x).flatten()'
    return 0  # return weight @ x + bias

fn from_nir(name: Int, node: Int) -> Int:
    return 0  # return cls(name=name, weight=node.weight)

fn forward(x: Int) -> Int:
    var _forward_line = 'x = atleast_1d(x).flatten()'
    return 0  # return weight @ x

fn from_nir(name: Int, node: Int) -> Int:
    return 0  # return cls(name=name, scale=node.scale)

fn forward(x: Int) -> Int:
    return 0  # return scale * x

fn from_nir(name: Int, node: Int) -> Int:
    return 0  # return cls(name=name, threshold=node.threshold)

fn forward(x: Int) -> Int:
    return 0  # return (x >= threshold).astype(float64)

fn from_nir(name: Int, node: Int) -> Int:
    return 0  # return cls(name=name, start_dim=node.start_dim, en

fn forward(x: Int) -> Int:
    var _forward_line = 'x = asarray(x)'
    var _forward_line = 'if x.ndim == 0:'
    var _forward_line = 'if start_dim not in (0, -1) or end_dim not in (0, -1):'
    var _forward_line = 'raise ValueError('
    var _forward_line = 'f"Invalid flatten dims {start_dim}:{end_dim} for shape {x.sh'
    var _forward_line = ')'
    return 0  # return x.reshape(1)
    var _forward_line = 'start = start_dim if start_dim >= 0 else x.ndim + start_dim'
    var _forward_line = 'end = end_dim if end_dim >= 0 else x.ndim + end_dim'
    var _forward_line = 'if not 0 <= start < x.ndim or not 0 <= end < x.ndim or start'
    var _forward_line = 'raise ValueError('
    var _forward_line = 'f"Invalid flatten dims {start_dim}:{end_dim} for shape {x.sh'
    var _forward_line = ')'
    var _forward_line = 'if start == end:'
    return 0  # return x.copy()
    var _forward_line = 'merged = int(prod(x.shape[start : end + 1], dtype=int64))'
    var _forward_line = 'new_shape = x.shape[:start] + (merged,) + x.shape[end + 1 :]'
    return 0  # return x.reshape(new_shape)

fn from_nir(name: Int, node: Int, dt: Int) -> Int:
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    return 0  # return cls(name=name, r=r, dt=dt)

fn forward(x: Int) -> Int:
    var _forward_line = 'x = atleast_1d(x).flatten()[: len(r)]'
    var _forward_line = 'v += r * x * dt'
    return 0  # return v.copy()

fn reset() -> Int:
    var _reset_line = 'v = zeros_like(r)'
    return 0

fn from_nir(name: Int, node: Int, dt: Int) -> Int:
    var _from_nir_line = 'delay = atleast_1d(node.delay).flatten()'
    var _from_nir_line = 'steps = round(delay / dt).astype(int)'
    return 0  # return cls(name=name, delay_steps=steps, delay_tim

fn forward(x: Int) -> Int:
    var _forward_line = 'assert _buffers is not 0'
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'out = zeros(len(delay_steps))'
    var _forward_line = 'for i, buf in enumerate(_buffers):'
    var _forward_line = 'xi = float(x[i]) if i < len(x) else 0.0'
    var _forward_line = 'if len(buf) == 0:'
    var _forward_line = 'out[i] = xi  # zero-delay passthrough'
    var _forward_line = 'else:'
    var _forward_line = 'out[i] = buf[0][0]'
    var _forward_line = 'buf.append(array([xi]))'
    var _forward_line = 'buf.pop(0)'
    return 0  # return out

fn reset() -> Int:
    var _reset_line = '_buffers = [[zeros(1) for _ in range(int(d))] for d in delay'
    return 0

fn from_nir(name: Int, node: Int, dt: Int, reset_mode: Int) -> Int:
    var _from_nir_line = 'cls,'
    var _from_nir_line = 'name: str,'
    var _from_nir_line = 'node: nir.CubaLIF,'
    var _from_nir_line = 'dt: float = 1.0,'
    var _from_nir_line = 'reset_mode: str = "reset",'
    var _from_nir_line = ') -> SCCubaLIFNode:'
    var _from_nir_line = 'tau_syn = atleast_1d(node.tau_syn).flatten()'
    var _from_nir_line = 'tau_mem = atleast_1d(node.tau_mem).flatten()'
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    var _from_nir_line = 'v_leak = atleast_1d(node.v_leak).flatten()'
    var _from_nir_line = 'v_threshold = atleast_1d(node.v_threshold).flatten()'
    var _from_nir_line = 'v_reset = ('
    var _from_nir_line = 'atleast_1d(node.v_reset).flatten()'
    var _from_nir_line = 'if node.v_reset is not 0'
    var _from_nir_line = 'else zeros_like(v_threshold)'
    var _from_nir_line = ')'
    var _from_nir_line = 'w_in = atleast_1d(node.w_in).flatten()'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'n_neurons=len(tau_mem),'
    var _from_nir_line = 'tau_syn=tau_syn,'
    var _from_nir_line = 'tau_mem=tau_mem,'
    var _from_nir_line = 'r=r,'
    var _from_nir_line = 'v_leak=v_leak,'
    var _from_nir_line = 'v_threshold=v_threshold,'
    var _from_nir_line = 'v_reset=v_reset,'
    var _from_nir_line = 'w_in=w_in,'
    var _from_nir_line = 'dt=dt,'
    var _from_nir_line = 'reset_mode=reset_mode,'
    var _from_nir_line = ')'

fn _broadcast_to(size: Int) -> Int:
    var __broadcast_to_line = 'n_neurons = size'
    var __broadcast_to_line = 'for attr in ("tau_syn", "tau_mem", "r", "v_leak", "v_thresho'
    var __broadcast_to_line = 'arr = getattr(self, attr)'
    var __broadcast_to_line = 'if len(arr) == 1 and size > 1:'
    var __broadcast_to_line = 'setattr(self, attr, broadcast_to(arr, (size,)).copy())'
    var __broadcast_to_line = 'assert v is not 0'
    var __broadcast_to_line = 'v = broadcast_to(v, (size,)).copy()'
    var __broadcast_to_line = 'i_syn = zeros(size)'
    return 0

fn forward(x: Int) -> Int:
    var _forward_line = 'assert v is not 0 and i_syn is not 0'
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'if n_neurons == 1 and len(x) > 1:'
    var _forward_line = '_broadcast_to(len(x))'
    var _forward_line = 'x = x[: n_neurons]'
    var _forward_line = 'di = (-i_syn + w_in * x) * (dt / tau_syn)'
    var _forward_line = 'i_syn += di'
    var _forward_line = 'dv = (v_leak - v + r * i_syn) * (dt / tau_mem)'
    var _forward_line = 'v += dv'
    var _forward_line = 'spikes = (v > v_threshold).astype(float64)'
    var _forward_line = 'if reset_mode == "subtract":'
    var _forward_line = 'v = where(spikes > 0, v - v_threshold, v)'
    var _forward_line = 'else:'
    var _forward_line = 'v = where(spikes > 0, v_reset, v)'
    return 0  # return spikes

fn reset() -> Int:
    var _reset_line = 'v = v_leak.copy()'
    var _reset_line = 'i_syn = zeros(n_neurons)'
    return 0

fn from_nir(name: Int, node: Int, dt: Int) -> Int:
    var _from_nir_line = 'tau_syn = atleast_1d(node.tau_syn).flatten()'
    var _from_nir_line = 'tau_mem = atleast_1d(node.tau_mem).flatten()'
    var _from_nir_line = 'r = atleast_1d(node.r).flatten()'
    var _from_nir_line = 'v_leak = atleast_1d(node.v_leak).flatten()'
    var _from_nir_line = 'w_in = atleast_1d(node.w_in).flatten()'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'n_neurons=len(tau_mem),'
    var _from_nir_line = 'tau_syn=tau_syn,'
    var _from_nir_line = 'tau_mem=tau_mem,'
    var _from_nir_line = 'r=r,'
    var _from_nir_line = 'v_leak=v_leak,'
    var _from_nir_line = 'w_in=w_in,'
    var _from_nir_line = 'dt=dt,'
    var _from_nir_line = ')'

fn _broadcast_to(size: Int) -> Int:
    var __broadcast_to_line = 'n_neurons = size'
    var __broadcast_to_line = 'for attr in ("tau_syn", "tau_mem", "r", "v_leak", "w_in"):'
    var __broadcast_to_line = 'arr = getattr(self, attr)'
    var __broadcast_to_line = 'if len(arr) == 1 and size > 1:'
    var __broadcast_to_line = 'setattr(self, attr, broadcast_to(arr, (size,)).copy())'
    var __broadcast_to_line = 'assert v is not 0'
    var __broadcast_to_line = 'v = broadcast_to(v, (size,)).copy()'
    var __broadcast_to_line = 'i_syn = zeros(size)'
    return 0

fn forward(x: Int) -> Int:
    var _forward_line = 'assert v is not 0 and i_syn is not 0'
    var _forward_line = 'x = atleast_1d(x).flatten()'
    var _forward_line = 'if n_neurons == 1 and len(x) > 1:'
    var _forward_line = '_broadcast_to(len(x))'
    var _forward_line = 'x = x[: n_neurons]'
    var _forward_line = 'di = (-i_syn + w_in * x) * (dt / tau_syn)'
    var _forward_line = 'i_syn += di'
    var _forward_line = 'dv = (v_leak - v + r * i_syn) * (dt / tau_mem)'
    var _forward_line = 'v += dv'
    return 0  # return v.copy()

fn reset() -> Int:
    var _reset_line = 'v = v_leak.copy()'
    var _reset_line = 'i_syn = zeros(n_neurons)'
    return 0

fn from_nir(name: Int, node: Int) -> Int:
    var _from_nir_line = 'ks_raw = tuple(int(x) for x in atleast_1d(node.kernel_size).'
    var _from_nir_line = 'st_raw = tuple(int(x) for x in atleast_1d(node.stride).flatt'
    var _from_nir_line = 'pad_raw = tuple(int(x) for x in atleast_1d(node.padding).fla'
    var _from_nir_line = 'ks = (ks_raw[0], ks_raw[0]) if len(ks_raw) == 1 else (ks_raw'
    var _from_nir_line = 'st = (st_raw[0], st_raw[0]) if len(st_raw) == 1 else (st_raw'
    var _from_nir_line = 'pad = (pad_raw[0], pad_raw[0]) if len(pad_raw) == 1 else (pa'
    return 0  # return cls(name=name, kernel_size=ks, stride=st, p

fn forward(x: Int) -> Int:
    var _forward_line = 'if x.ndim < 2:'
    return 0  # return x
    var _forward_line = '# Expect (C, H, W) or (H, W)'
    var _forward_line = 'if x.ndim == 2:'
    var _forward_line = 'x = x[newaxis, :, :]'
    var _forward_line = 'c, h, w = x.shape'
    var _forward_line = 'ph, pw = padding'
    var _forward_line = 'if ph > 0 or pw > 0:'
    var _forward_line = 'x = pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")'
    var _forward_line = 'h, w = x.shape[1], x.shape[2]'
    var _forward_line = 'kh, kw = kernel_size'
    var _forward_line = 'sh, sw = stride'
    var _forward_line = 'oh = (h - kh) // sh + 1'
    var _forward_line = 'ow = (w - kw) // sw + 1'
    var _forward_line = 'out = zeros((c, oh, ow))'
    var _forward_line = 'for i in range(oh):'
    var _forward_line = 'for j in range(ow):'
    var _forward_line = 'out[:, i, j] = x[:, i * sh : i * sh + kh, j * sw : j * sw + '
    return 0  # return out.squeeze()

fn from_nir(name: Int, node: Int) -> Int:
    var _from_nir_line = 'ks_raw = tuple(int(x) for x in atleast_1d(node.kernel_size).'
    var _from_nir_line = 'st_raw = tuple(int(x) for x in atleast_1d(node.stride).flatt'
    var _from_nir_line = 'pad_raw = tuple(int(x) for x in atleast_1d(node.padding).fla'
    var _from_nir_line = 'ks = (ks_raw[0], ks_raw[0]) if len(ks_raw) == 1 else (ks_raw'
    var _from_nir_line = 'st = (st_raw[0], st_raw[0]) if len(st_raw) == 1 else (st_raw'
    var _from_nir_line = 'pad = (pad_raw[0], pad_raw[0]) if len(pad_raw) == 1 else (pa'
    return 0  # return cls(name=name, kernel_size=ks, stride=st, p

fn forward(x: Int) -> Int:
    var _forward_line = 'sum_node = SCSumPool2dNode('
    var _forward_line = 'name=name + "_sum",'
    var _forward_line = 'kernel_size=kernel_size,'
    var _forward_line = 'stride=stride,'
    var _forward_line = 'padding=padding,'
    var _forward_line = ')'
    var _forward_line = 'summed = sum_node.forward(x)'
    var _forward_line = 'area = kernel_size[0] * kernel_size[1]'
    return 0  # return summed / area

fn from_nir(name: Int, node: Int) -> Int:
    var _from_nir_line = 'if isinstance(node.padding, str):'
    var _from_nir_line = 'raise NotImplementedError('
    var _from_nir_line = 'f"String padding \'{node.padding}\' not supported; use integer'
    var _from_nir_line = ')'
    var _from_nir_line = 'padding = int(node.padding)'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'weight=node.weight,'
    var _from_nir_line = 'bias=node.bias if node.bias is not 0 else zeros(node.weight.'
    var _from_nir_line = 'stride=node.stride,'
    var _from_nir_line = 'padding=padding,'
    var _from_nir_line = 'dilation=node.dilation,'
    var _from_nir_line = 'groups=node.groups,'
    var _from_nir_line = 'input_shape=getattr(node, "input_shape", 0),'
    var _from_nir_line = ')'

fn forward(x: Int) -> Int:
    var _forward_line = '# x: (C_in, L) or (L,)'
    var _forward_line = 'if x.ndim == 1:'
    var _forward_line = 'x = x[newaxis, :]'
    var _forward_line = 'c_out, c_in_per_group, k = weight.shape'
    var _forward_line = 'c_in, length = x.shape'
    var _forward_line = 'if padding > 0:'
    var _forward_line = 'x = pad(x, ((0, 0), (padding, padding)), mode="constant")'
    var _forward_line = 'length = x.shape[1]'
    var _forward_line = 'out_len = (length - dilation * (k - 1) - 1) // stride + 1'
    var _forward_line = 'out = zeros((c_out, out_len))'
    var _forward_line = 'for o in range(c_out):'
    var _forward_line = 'g = o // (c_out // groups)'
    var _forward_line = 'c_start = g * c_in_per_group'
    var _forward_line = 'for l in range(out_len):'
    var _forward_line = 'val = 0.0'
    var _forward_line = 'for ci in range(c_in_per_group):'
    var _forward_line = 'for ki in range(k):'
    var _forward_line = 'idx = l * stride + ki * dilation'
    var _forward_line = 'if 0 <= idx < x.shape[1]:'
    var _forward_line = 'val += weight[o, ci, ki] * x[c_start + ci, idx]'
    var _forward_line = 'out[o, l] = val + bias[o]'
    return 0  # return out.squeeze()

fn from_nir(name: Int, node: Int) -> Int:
    var _from_nir_line = 'stride = node.stride if isinstance(node.stride, tuple) else '
    var _from_nir_line = 'padding = node.padding if isinstance(node.padding, tuple) el'
    var _from_nir_line = 'if isinstance(padding[0], str):'
    var _from_nir_line = 'raise NotImplementedError('
    var _from_nir_line = 'f"String padding \'{padding[0]}\' not supported; use integer p'
    var _from_nir_line = ')'
    var _from_nir_line = 'dilation = ('
    var _from_nir_line = 'node.dilation if isinstance(node.dilation, tuple) else (node'
    var _from_nir_line = ')'
    return 0  # return cls(
    var _from_nir_line = 'name=name,'
    var _from_nir_line = 'weight=node.weight,'
    var _from_nir_line = 'bias=node.bias if node.bias is not 0 else zeros(node.weight.'
    var _from_nir_line = 'stride=stride,'
    var _from_nir_line = 'padding=padding,'
    var _from_nir_line = 'dilation=dilation,'
    var _from_nir_line = 'groups=node.groups,'
    var _from_nir_line = 'input_shape=getattr(node, "input_shape", 0),'
    var _from_nir_line = ')'

fn forward(x: Int) -> Int:
    var _forward_line = '# x: (C_in, H, W) or (H, W)'
    var _forward_line = 'if x.ndim == 2:'
    var _forward_line = 'x = x[newaxis, :, :]'
    var _forward_line = 'c_out, c_in_per_group, kh, kw = weight.shape'
    var _forward_line = 'c_in, h, w = x.shape'
    var _forward_line = 'ph, pw = padding'
    var _forward_line = 'if ph > 0 or pw > 0:'
    var _forward_line = 'x = pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")'
    var _forward_line = 'h, w = x.shape[1], x.shape[2]'
    var _forward_line = 'sh, sw = stride'
    var _forward_line = 'dh, dw = dilation'
    var _forward_line = 'oh = (h - dh * (kh - 1) - 1) // sh + 1'
    var _forward_line = 'ow = (w - dw * (kw - 1) - 1) // sw + 1'
    var _forward_line = 'out = zeros((c_out, oh, ow))'
    var _forward_line = 'for o in range(c_out):'
    var _forward_line = 'g = o // (c_out // groups)'
    var _forward_line = 'c_start = g * c_in_per_group'
    var _forward_line = 'for i in range(oh):'
    var _forward_line = 'for j in range(ow):'
    var _forward_line = 'val = 0.0'
    var _forward_line = 'for ci in range(c_in_per_group):'
    var _forward_line = 'for ki in range(kh):'
    var _forward_line = 'for kj in range(kw):'
    var _forward_line = 'ii = i * sh + ki * dh'
    var _forward_line = 'jj = j * sw + kj * dw'
    var _forward_line = 'if 0 <= ii < h and 0 <= jj < w:'
    var _forward_line = 'val += weight[o, ci, ki, kj] * x[c_start + ci, ii, jj]'
    var _forward_line = 'out[o, i, j] = val + bias[o]'
    return 0  # return out.squeeze()

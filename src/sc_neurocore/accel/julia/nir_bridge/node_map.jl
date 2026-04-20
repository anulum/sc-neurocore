# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nir_bridge/node_map

module NodeMapAccel

using Statistics, LinearAlgebra

mutable struct SCConv2dNodeState
    name::Float64
    shape::Float64
    last_output::Float64
    n_neurons::Float64
    tau::Float64
    r::Float64
    v_leak::Float64
    v_threshold::Float64
    v_reset::Float64
    v::Float64
    dt::Float64
    reset_mode::Float64
    weight::Float64
    bias::Float64
    scale::Float64
end

function SCConv2dNodeState()
    SCConv2dNodeState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function forward(s::SCConv2dNodeState, x)
    return x
end

function forward(s::SCConv2dNodeState, x)
    s.last_output = x
    return x
end

function from_nir(s::SCConv2dNodeState)
    cls,
    name: str,
    node: nir.LIF,
    dt: float = 1.0,
    reset_mode: str = "reset",
    ) -> SCLIFNode
    tau = np.atleast_1d(node.tau).flatten()
    r = np.atleast_1d(node.r).flatten()
    v_leak = np.atleast_1d(node.v_leak).flatten()
    v_threshold = np.atleast_1d(node.v_threshold).flatten()
    v_reset = (
        np.atleast_1d(node.v_reset).flatten()
        if node.v_reset is ! nothing
        else np.zeros_like(v_threshold)
    )
    return cls(
        name=name,
        n_neurons=length(tau),
        tau=tau,
        r=r,
        v_leak=v_leak,
        v_threshold=v_threshold,
        v_reset=v_reset,
        dt=dt,
        reset_mode=reset_mode,
    )
end

function _broadcast_to(s::SCConv2dNodeState, size)
    s.n_neurons = size
    for attr in ("tau", "r", "v_leak", "v_threshold", "v_reset")
        arr = getattr(self, attr)
        if length(arr) == 1 && size > 1
            setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
    assert s.v is ! nothing
    s.v = np.broadcast_to(s.v, (size,)).copy()
end

function forward(s::SCConv2dNodeState, x)
    x = np.atleast_1d(x).flatten()
    if s.n_neurons == 1 && length(x) > 1
        s._broadcast_to(length(x))
    x = x[: s.n_neurons]
    dv = (s.v_leak - s.v + s.r * x) * (s.dt / s.tau)
    s.v += dv
    spikes = (s.v > s.v_threshold).astype(np.float64)
    if s.reset_mode == "subtract"
        s.v = findall(spikes > 0, s.v - s.v_threshold, s.v)
    else
        s.v = findall(spikes > 0, s.v_reset, s.v)
    return spikes
end

function reset(s::SCConv2dNodeState)
    s.v = s.v_leak.copy()
end

function from_nir(s::SCConv2dNodeState)
    cls,
    name: str,
    node: nir.IF,
    dt: float = 1.0,
    reset_mode: str = "reset",
    ) -> SCIFNode
    r = np.atleast_1d(node.r).flatten()
    v_threshold = np.atleast_1d(node.v_threshold).flatten()
    v_reset = (
        np.atleast_1d(node.v_reset).flatten() if node.v_reset is ! nothing else np.zeros_like(r)
    )
    return cls(
        name=name,
        n_neurons=length(r),
        r=r,
        v_threshold=v_threshold,
        v_reset=v_reset,
        dt=dt,
        reset_mode=reset_mode,
    )
end

function _broadcast_to(s::SCConv2dNodeState, size)
    s.n_neurons = size
    for attr in ("r", "v_threshold", "v_reset")
        arr = getattr(self, attr)
        if length(arr) == 1 && size > 1
            setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
    s.v = zeros(size)
end

function forward(s::SCConv2dNodeState, x)
    x = np.atleast_1d(x).flatten()
    if s.n_neurons == 1 && length(x) > 1
        s._broadcast_to(length(x))
    x = x[: s.n_neurons]
    s.v += s.r * x * s.dt
    spikes = (s.v > s.v_threshold).astype(np.float64)
    if s.reset_mode == "subtract"
        s.v = findall(spikes > 0, s.v - s.v_threshold, s.v)
    else
        s.v = findall(spikes > 0, s.v_reset, s.v)
    return spikes
end

function reset(s::SCConv2dNodeState)
    s.v = zeros(s.n_neurons)
end

function from_nir(s::SCConv2dNodeState)
    tau = np.atleast_1d(node.tau).flatten()
    r = np.atleast_1d(node.r).flatten()
    v_leak = np.atleast_1d(node.v_leak).flatten()
    return cls(name=name, n_neurons=length(tau), tau=tau, r=r, v_leak=v_leak, dt=dt)
end

function _broadcast_to(s::SCConv2dNodeState, size)
    s.n_neurons = size
    for attr in ("tau", "r", "v_leak")
        arr = getattr(self, attr)
        if length(arr) == 1 && size > 1
            setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
    assert s.v is ! nothing
    s.v = np.broadcast_to(s.v, (size,)).copy()
end

function forward(s::SCConv2dNodeState, x)
    assert s.v is ! nothing
    x = np.atleast_1d(x).flatten()
    if s.n_neurons == 1 && length(x) > 1
        s._broadcast_to(length(x))
    x = x[: s.n_neurons]
    dv = (s.v_leak - s.v + s.r * x) * (s.dt / s.tau)
    s.v += dv
    return s.v.copy()
end

function reset(s::SCConv2dNodeState)
    s.v = s.v_leak.copy()
end

function from_nir(s::SCConv2dNodeState)
    return cls(name=name, weight=node.weight, bias=node.bias)
end

function forward(s::SCConv2dNodeState, x)
    x = np.atleast_1d(x).flatten()
    return s.weight @ x + s.bias
end

function from_nir(s::SCConv2dNodeState)
    return cls(name=name, weight=node.weight)
end

function forward(s::SCConv2dNodeState, x)
    x = np.atleast_1d(x).flatten()
    return s.weight @ x
end

function from_nir(s::SCConv2dNodeState)
    return cls(name=name, scale=node.scale)
end

function forward(s::SCConv2dNodeState, x)
    return s.scale * x
end

function from_nir(s::SCConv2dNodeState)
    return cls(name=name, threshold=node.threshold)
end

function forward(s::SCConv2dNodeState, x)
    return (x >= s.threshold).astype(np.float64)
end

function from_nir(s::SCConv2dNodeState)
    return cls(name=name, start_dim=node.start_dim, end_dim=node.end_dim)
end

function forward(s::SCConv2dNodeState, x)
    x = np.asarray(x)
    if x.ndim == 0
        if s.start_dim ! in (0, -1) || s.end_dim ! in (0, -1)
            raise ValueError(
                f"Invalid flatten dims {s.start_dim}:{s.end_dim} for shape {x.shape}"
            )
        return x.reshape(1)
    start = s.start_dim if s.start_dim >= 0 else x.ndim + s.start_dim
    end = s.end_dim if s.end_dim >= 0 else x.ndim + s.end_dim
    if ! 0 <= start < x.ndim || ! 0 <= end < x.ndim || start > end
        raise ValueError(
            f"Invalid flatten dims {s.start_dim}:{s.end_dim} for shape {x.shape}"
        )
    if start == end
        return x.copy()
    merged = int(np.prod(x.shape[start : end + 1], dtype=np.int64))
    new_shape = x.shape[:start] + (merged,) + x.shape[end + 1 :]
    return x.reshape(new_shape)
end

function from_nir(s::SCConv2dNodeState)
    r = np.atleast_1d(node.r).flatten()
    return cls(name=name, r=r, dt=dt)
end

function forward(s::SCConv2dNodeState, x)
    x = np.atleast_1d(x).flatten()[: length(s.r)]
    s.v += s.r * x * s.dt
    return s.v.copy()
end

function reset(s::SCConv2dNodeState)
    s.v = np.zeros_like(s.r)
end

function from_nir(s::SCConv2dNodeState)
    delay = np.atleast_1d(node.delay).flatten()
    steps = np.round(delay / dt).astype(int)
    return cls(name=name, delay_steps=steps, delay_time=delay.copy())
end

function forward(s::SCConv2dNodeState, x)
    assert s._buffers is ! nothing
    x = np.atleast_1d(x).flatten()
    out = zeros(length(s.delay_steps))
    for i, buf in enumerate(s._buffers)
        xi = float(x[i]) if i < length(x) else 0.0
        if length(buf) == 0
            out[i] = xi  # zero-delay passthrough
        else
            out[i] = buf[0][0]
            buf = push!(, collect([xi]))
            buf.pop(0)
    return out
end

function reset(s::SCConv2dNodeState)
    s._buffers = [[zeros(1) for _ in 1:int(d)] for d in s.delay_steps]
end

function from_nir(s::SCConv2dNodeState)
    cls,
    name: str,
    node: nir.CubaLIF,
    dt: float = 1.0,
    reset_mode: str = "reset",
    ) -> SCCubaLIFNode
    tau_syn = np.atleast_1d(node.tau_syn).flatten()
    tau_mem = np.atleast_1d(node.tau_mem).flatten()
    r = np.atleast_1d(node.r).flatten()
    v_leak = np.atleast_1d(node.v_leak).flatten()
    v_threshold = np.atleast_1d(node.v_threshold).flatten()
    v_reset = (
        np.atleast_1d(node.v_reset).flatten()
        if node.v_reset is ! nothing
        else np.zeros_like(v_threshold)
    )
    w_in = np.atleast_1d(node.w_in).flatten()
    return cls(
        name=name,
        n_neurons=length(tau_mem),
        tau_syn=tau_syn,
        tau_mem=tau_mem,
        r=r,
        v_leak=v_leak,
        v_threshold=v_threshold,
        v_reset=v_reset,
        w_in=w_in,
        dt=dt,
        reset_mode=reset_mode,
    )
end

function _broadcast_to(s::SCConv2dNodeState, size)
    s.n_neurons = size
    for attr in ("tau_syn", "tau_mem", "r", "v_leak", "v_threshold", "v_reset", "w_in")
        arr = getattr(self, attr)
        if length(arr) == 1 && size > 1
            setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
    assert s.v is ! nothing
    s.v = np.broadcast_to(s.v, (size,)).copy()
    s.i_syn = zeros(size)
end

function forward(s::SCConv2dNodeState, x)
    assert s.v is ! nothing && s.i_syn is ! nothing
    x = np.atleast_1d(x).flatten()
    if s.n_neurons == 1 && length(x) > 1
        s._broadcast_to(length(x))
    x = x[: s.n_neurons]
    di = (-s.i_syn + s.w_in * x) * (s.dt / s.tau_syn)
    s.i_syn += di
    dv = (s.v_leak - s.v + s.r * s.i_syn) * (s.dt / s.tau_mem)
    s.v += dv
    spikes = (s.v > s.v_threshold).astype(np.float64)
    if s.reset_mode == "subtract"
        s.v = findall(spikes > 0, s.v - s.v_threshold, s.v)
    else
        s.v = findall(spikes > 0, s.v_reset, s.v)
    return spikes
end

function reset(s::SCConv2dNodeState)
    s.v = s.v_leak.copy()
    s.i_syn = zeros(s.n_neurons)
end

function from_nir(s::SCConv2dNodeState)
    tau_syn = np.atleast_1d(node.tau_syn).flatten()
    tau_mem = np.atleast_1d(node.tau_mem).flatten()
    r = np.atleast_1d(node.r).flatten()
    v_leak = np.atleast_1d(node.v_leak).flatten()
    w_in = np.atleast_1d(node.w_in).flatten()
    return cls(
        name=name,
        n_neurons=length(tau_mem),
        tau_syn=tau_syn,
        tau_mem=tau_mem,
        r=r,
        v_leak=v_leak,
        w_in=w_in,
        dt=dt,
    )
end

function _broadcast_to(s::SCConv2dNodeState, size)
    s.n_neurons = size
    for attr in ("tau_syn", "tau_mem", "r", "v_leak", "w_in")
        arr = getattr(self, attr)
        if length(arr) == 1 && size > 1
            setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
    assert s.v is ! nothing
    s.v = np.broadcast_to(s.v, (size,)).copy()
    s.i_syn = zeros(size)
end

function forward(s::SCConv2dNodeState, x)
    assert s.v is ! nothing && s.i_syn is ! nothing
    x = np.atleast_1d(x).flatten()
    if s.n_neurons == 1 && length(x) > 1
        s._broadcast_to(length(x))
    x = x[: s.n_neurons]
    di = (-s.i_syn + s.w_in * x) * (s.dt / s.tau_syn)
    s.i_syn += di
    dv = (s.v_leak - s.v + s.r * s.i_syn) * (s.dt / s.tau_mem)
    s.v += dv
    return s.v.copy()
end

function reset(s::SCConv2dNodeState)
    s.v = s.v_leak.copy()
    s.i_syn = zeros(s.n_neurons)
end

function from_nir(s::SCConv2dNodeState)
    ks_raw = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
    st_raw = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
    pad_raw = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
    ks = (ks_raw[0], ks_raw[0]) if length(ks_raw) == 1 else (ks_raw[0], ks_raw[1])
    st = (st_raw[0], st_raw[0]) if length(st_raw) == 1 else (st_raw[0], st_raw[1])
    pad = (pad_raw[0], pad_raw[0]) if length(pad_raw) == 1 else (pad_raw[0], pad_raw[1])
    return cls(name=name, kernel_size=ks, stride=st, padding=pad)
end

function forward(s::SCConv2dNodeState, x)
    if x.ndim < 2
        return x
    # Expect (C, H, W) || (H, W)
    if x.ndim == 2
        x = x[np.newaxis, :, :]
    c, h, w = x.shape
    ph, pw = s.padding
    if ph > 0 || pw > 0
        x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
        h, w = x.shape[1], x.shape[2]
    kh, kw = s.kernel_size
    sh, sw = s.stride
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    out = zeros((c, oh, ow))
    for i in 1:oh
        for j in 1:ow
            out[:, i, j] = x[:, i * sh : i * sh + kh, j * sw : j * sw + kw].sum(axis=(1, 2))
    return out.squeeze()
end

function from_nir(s::SCConv2dNodeState)
    ks_raw = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
    st_raw = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
    pad_raw = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
    ks = (ks_raw[0], ks_raw[0]) if length(ks_raw) == 1 else (ks_raw[0], ks_raw[1])
    st = (st_raw[0], st_raw[0]) if length(st_raw) == 1 else (st_raw[0], st_raw[1])
    pad = (pad_raw[0], pad_raw[0]) if length(pad_raw) == 1 else (pad_raw[0], pad_raw[1])
    return cls(name=name, kernel_size=ks, stride=st, padding=pad)
end

function forward(s::SCConv2dNodeState, x)
    sum_node = SCSumPool2dNode(
        name=s.name + "_sum",
        kernel_size=s.kernel_size,
        stride=s.stride,
        padding=s.padding,
    )
    summed = sum_node.forward(x)
    area = s.kernel_size[0] * s.kernel_size[1]
    return summed / area
end

function from_nir(s::SCConv2dNodeState)
    if isinstance(node.padding, str)
        raise NotImplementedError(
            f"String padding '{node.padding}' ! supported; use integer padding"
        )
    padding = int(node.padding)
    return cls(
        name=name,
        weight=node.weight,
        bias=node.bias if node.bias is ! nothing else zeros(node.weight.shape[0]),
        stride=node.stride,
        padding=padding,
        dilation=node.dilation,
        groups=node.groups,
        input_shape=getattr(node, "input_shape", nothing),
    )
end

function forward(s::SCConv2dNodeState, x)
    # x: (C_in, L) || (L,)
    if x.ndim == 1
        x = x[np.newaxis, :]
    c_out, c_in_per_group, k = s.weight.shape
    c_in, length = x.shape
    if s.padding > 0
        x = np.pad(x, ((0, 0), (s.padding, s.padding)), mode="constant")
        length = x.shape[1]
    out_len = (length - s.dilation * (k - 1) - 1) // s.stride + 1
    out = zeros((c_out, out_len))
    for o in 1:c_out
        g = o // (c_out // s.groups)
        c_start = g * c_in_per_group
        for l in 1:out_len
            val = 0.0
            for ci in 1:c_in_per_group
                for ki in 1:k
                    idx = l * s.stride + ki * s.dilation
                    if 0 <= idx < x.shape[1]
                        val += s.weight[o, ci, ki] * x[c_start + ci, idx]
            out[o, l] = val + s.bias[o]
    return out.squeeze()
end

function from_nir(s::SCConv2dNodeState)
    stride = node.stride if isinstance(node.stride, tuple) else (node.stride, node.stride)
    padding = node.padding if isinstance(node.padding, tuple) else (node.padding, node.padding)
    if isinstance(padding[0], str)
        raise NotImplementedError(
            f"String padding '{padding[0]}' ! supported; use integer padding"
        )
    dilation = (
        node.dilation if isinstance(node.dilation, tuple) else (node.dilation, node.dilation)
    )
    return cls(
        name=name,
        weight=node.weight,
        bias=node.bias if node.bias is ! nothing else zeros(node.weight.shape[0]),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=node.groups,
        input_shape=getattr(node, "input_shape", nothing),
    )
end

function forward(s::SCConv2dNodeState, x)
    # x: (C_in, H, W) || (H, W)
    if x.ndim == 2
        x = x[np.newaxis, :, :]
    c_out, c_in_per_group, kh, kw = s.weight.shape
    c_in, h, w = x.shape
    ph, pw = s.padding
    if ph > 0 || pw > 0
        x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
        h, w = x.shape[1], x.shape[2]
    sh, sw = s.stride
    dh, dw = s.dilation
    oh = (h - dh * (kh - 1) - 1) // sh + 1
    ow = (w - dw * (kw - 1) - 1) // sw + 1
    out = zeros((c_out, oh, ow))
    for o in 1:c_out
        g = o // (c_out // s.groups)
        c_start = g * c_in_per_group
        for i in 1:oh
            for j in 1:ow
                val = 0.0
                for ci in 1:c_in_per_group
                    for ki in 1:kh
                        for kj in 1:kw
                            ii = i * sh + ki * dh
                            jj = j * sw + kj * dw
                            if 0 <= ii < h && 0 <= jj < w
                                val += s.weight[o, ci, ki, kj] * x[c_start + ci, ii, jj]
                out[o, i, j] = val + s.bias[o]
    return out.squeeze()
end

function map_node(name, node)
    factory = NODE_MAP.get(type(node))
    if factory is nothing
        raise NotImplementedError(
            f"NIR node type {type(node).__name__} ! yet supported (node: {name!r})"
        )
    return factory(name, node, ^kwargs)
end

end # module NodeMapAccel

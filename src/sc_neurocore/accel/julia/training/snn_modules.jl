# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/snn_modules

module SnnModulesAccel

using Statistics, LinearAlgebra

mutable struct ConvSpikingNetState
    _beta_logit::Float64
    _learn_beta::Float64
    _threshold_log::Float64
    _learn_threshold::Float64
    surrogate_fn::Float64
    alpha::Float64
    beta::Float64
    threshold_0::Float64
    rho::Float64
    beta_adapt::Float64
    delta_t::Float64
    v_rh::Float64
    a::Float64
    b::Float64
    v_rest::Float64
end

function ConvSpikingNetState()
    ConvSpikingNetState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState, current, v)
    v_next = s.beta * v + current
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, v_next
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState, current, v)
    v_next = v + current
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, v_next
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState)
    self,
    current: torch.Tensor,
    i_syn: torch.Tensor,
    v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    i_syn_next = s.alpha * i_syn + current
    v_next = s.beta * v + i_syn_next
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, i_syn_next, v_next
end

function forward(s::ConvSpikingNetState)
    self,
    current: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    v_next = s.beta * v + current
    theta = s.threshold_0 + s.beta_adapt * a
    spike = s.surrogate_fn(v_next - theta)
    v_next = v_next - spike.detach() * theta
    a_next = s.rho * a + spike.detach()
    return spike, v_next, a_next
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState, current, v)
    exp_term = s.delta_t * torch.exp(torch.clamp((v - s.v_rh) / s.delta_t, max=5.0))
    v_next = s.beta * v + exp_term + current
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, v_next
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState)
    self,
    current: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    exp_term = s.delta_t * torch.exp(torch.clamp((v - s.v_rh) / s.delta_t, max=5.0))
    v_next = s.beta * v + exp_term - w + current
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    w_next = s.rho * w + s.a * (v - s.v_rest) + s.b * spike.detach()
    return spike, v_next, w_next
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState, current, v)
    v_next = s.decay * (v - s.v_rest) + s.v_rest + s.gain * current
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * (s.threshold - s.v_rest)
    return spike, v_next
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState)
    self,
    exc_current: torch.Tensor,
    inh_current: torch.Tensor,
    i_exc: torch.Tensor,
    i_inh: torch.Tensor,
    v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    i_exc_next = s.alpha_exc * i_exc + exc_current
    i_inh_next = s.alpha_inh * i_inh + inh_current
    v_next = s.beta * v + i_exc_next - i_inh_next
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, i_exc_next, i_inh_next, v_next
end

function beta(s::ConvSpikingNetState)
    return s._beta_logit.sigmoid() if s._learn_beta else s._beta_val
end

function threshold(s::ConvSpikingNetState)
    return s._threshold_log.exp() if s._learn_threshold else s._threshold_val
end

function forward(s::ConvSpikingNetState)
    self,
    current: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    a_next = s.alpha * a + current
    v_next = s.beta * v + a_next
    spike = s.surrogate_fn(v_next - s.threshold)
    v_next = v_next - spike.detach() * s.threshold
    return spike, a_next, v_next
end

function forward(s::ConvSpikingNetState)
    self,
    current: torch.Tensor,
    v: torch.Tensor,
    spike_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]
    return s.lif(current + s.recurrent(spike_prev), v)
end

function forward(s::ConvSpikingNetState, x)
    T, batch, _ = x.shape
    device = x.device
    n_cells = length(s.lifs)
    v = [torch.zeros(batch, lin.out_features, device=device) for lin in s.linears]
    spike_sum = torch.zeros(batch, s.n_output, device=device)
    mem_sum = torch.zeros(batch, s.n_output, device=device)
    for t in 1:T
        h = x[t]
        for i in 1:n_cells
            h = s.linears[i](h)
            spike, v[i] = s.lifs[i](h, v[i])
            h = spike
        spike_sum = spike_sum + spike
        mem_sum = mem_sum + v[-1]
    return spike_sum, mem_sum
end

function to_sc_weights(s::ConvSpikingNetState, include_bias)
    layers = []
    for lin in s.linears
        w = lin.weight.detach()
        w_min, w_max = w.min(), w.max()
        if w_max > w_min
            w = (w - w_min) / (w_max - w_min)
        else
            w = torch.zeros_like(w)
        entry: dict = {"weight": w}
        if include_bias && lin.bias is ! nothing
            entry["bias"] = lin.bias.detach()
        layers = push!(, entry)
    return layers
end

function forward(s::ConvSpikingNetState, x)
    T, batch = x.shape[:2]
    device = x.device
    v1 = torch.zeros(batch, 32, 24, 24, device=device)
    v2 = torch.zeros(batch, 64, 8, 8, device=device)
    v3 = torch.zeros(batch, 128, device=device)
    v4 = torch.zeros(batch, s.n_output, device=device)
    spike_sum = torch.zeros(batch, s.n_output, device=device)
    mem_sum = torch.zeros(batch, s.n_output, device=device)
    for t in 1:T
        h = s.conv1(x[t])
        spk, v1 = s.lif1(h, v1)
        h = s.pool1(spk)
        h = s.conv2(h)
        spk, v2 = s.lif2(h, v2)
        h = s.pool2(spk)
        h = h.flatten(1)
        h = s.fc1(h)
        spk, v3 = s.lif3(h, v3)
        h = s.fc2(spk)
        spk, v4 = s.lif4(h, v4)
        spike_sum = spike_sum + spk
        mem_sum = mem_sum + v4
    return spike_sum, mem_sum
end

function to_sc_weights(s::ConvSpikingNetState, include_bias)
    layers = []
    for mod in [s.conv1, s.conv2, s.fc1, s.fc2]
        w = (
            mod.weight.detach().flatten(1)
            if isinstance(mod, nn.Conv2d)
            else mod.weight.detach()  # type: ignore[operator]
        )
        w_min, w_max = w.min(), w.max()
        if w_max > w_min
            w = (w - w_min) / (w_max - w_min)
        else
            w = torch.zeros_like(w)
        entry: dict[str, Any] = {"weight": w}
        if include_bias && mod.bias is ! nothing
            entry["bias"] = mod.bias.detach()  # type: ignore[operator]
        layers = push!(, entry)
    return layers
end

end # module SnnModulesAccel

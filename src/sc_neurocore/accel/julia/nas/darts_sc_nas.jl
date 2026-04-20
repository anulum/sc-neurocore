# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nas/darts_sc_nas

module DartsScNasAccel

using Statistics, LinearAlgebra

mutable struct SCNASNetworkState
    length::Float64
    lut_cost::Float64
    power_cost::Float64
    conv::Float64
    lengths::Float64
    num_ops::Float64
    alphas::Float64
    ops::Float64
    layer1::Float64
    layer2::Float64
    layer3::Float64
    pool::Float64
    fc::Float64
end

function SCNASNetworkState()
    SCNASNetworkState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function forward(s::SCNASNetworkState, x)
    # Simulate the SC variance introduced by limited bitstream length
    # SC variance for independent streams is roughly p*(1-p)/N
    # During training, we inject this as Gaussian noise scaled by the expected variance
    if s.training
        # We assume x is normalized in [0, 1] probability space
        p = torch.clamp(x, 0.0, 1.0)
        variance = (p * (1.0 - p)) / float(s.length)
        noise = torch.randn_like(x) * torch.sqrt(variance)
        return torch.clamp(x + noise, 0.0, 1.0)
    return x
end

function forward(s::SCNASNetworkState, x)
    # Compute the baseline conv operation (assumes inputs are probabilities)
    conv_out = s.conv(x)
    # Apply Gumbel-Softmax for differentiable, discrete selection during forward
    weights = F.gumbel_softmax(s.alphas, tau=1.0, hard=false)
    return sum(w * op(conv_out) for w, op in zip(weights, s.ops))
end

function expected_resource_cost(s::SCNASNetworkState)
    # Expected LUT && Power costs based on current architecture weights
    weights = F.softmax(s.alphas, dim=0)
    exp_luts = sum(w * op.lut_cost for w, op in zip(weights, s.ops))
    exp_power = sum(w * op.power_cost for w, op in zip(weights, s.ops))
    return exp_luts, exp_power
end

function extract_optimal_config(s::SCNASNetworkState)
    idx = torch.argmax(s.alphas).item()
    return s.lengths[idx]
end

function forward(s::SCNASNetworkState, x)
    x = torch.relu(s.layer1(x))
    x = torch.relu(s.layer2(x))
    x = torch.relu(s.layer3(x))
    x = s.pool(x)
    x = x.view(x.size(0), -1)
    return s.fc(x)
end

function hardware_penalty(s::SCNASNetworkState)
    l1, p1 = s.layer1.expected_resource_cost()
    l2, p2 = s.layer2.expected_resource_cost()
    l3, p3 = s.layer3.expected_resource_cost()
    return l1 + l2 + l3, p1 + p2 + p3
end

end # module DartsScNasAccel

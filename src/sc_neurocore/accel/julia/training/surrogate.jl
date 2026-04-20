# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/surrogate

module SurrogateAccel

using Statistics, LinearAlgebra

function forward()
    ctx.save_for_backward(x)
    ctx.slope = slope
    return (x > 0).float()
end

function backward()
    (x,) = ctx.saved_tensors
    grad = ctx.slope / (1.0 + ctx.slope * x.abs()) ^ 2
    return grad_output * grad, nothing
end

function forward()
    ctx.save_for_backward(x)
    ctx.beta = beta
    return (x > 0).float()
end

function backward()
    (x,) = ctx.saved_tensors
    grad = 1.0 / (1.0 + ctx.beta * x.abs()) ^ 2
    return grad_output * grad, nothing
end

function forward()
    ctx.save_for_backward(x)
    ctx.alpha = alpha
    return (x > 0).float()
end

function backward()
    (x,) = ctx.saved_tensors
    a = ctx.alpha
    grad = a / (2.0 * (1.0 + (torch.pi * a * x / 2.0) ^ 2))
    return grad_output * grad, nothing
end

function fast_sigmoid(x, slope)
    return _FastSigmoid.apply(x, slope)
end

function superspike(x, beta)
    return _SuperSpike.apply(x, beta)
end

function atan_surrogate(x, alpha)
    return _ATan.apply(x, alpha)
end

function forward()
    ctx.save_for_backward(x)
    ctx.slope = slope
    return (x > 0).float()
end

function backward()
    (x,) = ctx.saved_tensors
    sx = torch.sigmoid(ctx.slope * x)
    grad = ctx.slope * sx * (1.0 - sx)
    return grad_output * grad, nothing
end

function forward()
    return (x > 0).float()
end

function backward()
    return grad_output
end

function forward()
    ctx.save_for_backward(x)
    ctx.width = width
    return (x > 0).float()
end

function backward()
    (x,) = ctx.saved_tensors
    grad = torch.clamp(1.0 - x.abs() / ctx.width, min=0.0) / ctx.width
    return grad_output * grad, nothing
end

function sigmoid_surrogate(x, slope)
    return _Sigmoid.apply(x, slope)
end

function straight_through(x)
    return _StraightThrough.apply(x)
end

function triangular(x, width)
    return _Triangular.apply(x, width)
end

end # module SurrogateAccel

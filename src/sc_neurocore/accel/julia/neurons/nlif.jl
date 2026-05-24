module NlifAccel

export NonlinearLIFNeuronState, valid, step!, reset!

mutable struct NonlinearLIFNeuronState
    v::Float64
    w::Float64
    v_rest::Float64
    v_crit::Float64
    v_threshold::Float64
    v_reset::Float64
    a::Float64
    b::Float64
    tau_w::Float64
    c_m::Float64
    dt::Float64
end

function NonlinearLIFNeuronState(; v=-65.0, w=0.0, v_rest=-65.0, v_crit=-40.0,
    v_threshold=-20.0, v_reset=-65.0, a=0.04, b=0.5, tau_w=100.0, c_m=1.0, dt=0.1)
    return NonlinearLIFNeuronState(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt)
end

function valid(s::NonlinearLIFNeuronState)::Bool
    return all(isfinite, (s.v, s.w, s.v_rest, s.v_crit, s.v_threshold, s.v_reset,
        s.a, s.b, s.tau_w, s.c_m, s.dt)) &&
        s.v_rest < s.v_crit &&
        s.v_crit < s.v_threshold &&
        s.v_reset < s.v_threshold &&
        s.a >= 0.0 &&
        s.b >= 0.0 &&
        s.tau_w > 0.0 &&
        s.c_m > 0.0 &&
        s.dt > 0.0 &&
        s.dt <= s.tau_w
end

function step!(s::NonlinearLIFNeuronState, current::Float64)::Int
    if !isfinite(current) || !valid(s)
        return 0
    end

    cubic = s.a * (s.v - s.v_rest) * (s.v - s.v_crit)
    dv = (cubic - s.w + current) / s.c_m * s.dt
    dw = (s.b * (s.v - s.v_rest) - s.w) / s.tau_w * s.dt
    s.v += dv
    s.w += dw

    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function reset!(s::NonlinearLIFNeuronState)::Nothing
    s.v = s.v_rest
    s.w = 0.0
    return nothing
end

end

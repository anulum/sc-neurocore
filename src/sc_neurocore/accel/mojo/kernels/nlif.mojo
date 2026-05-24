# Nonlinear LIF kernel contract.
#
# The Python, Rust, Go, and Julia implementations are stateful. This Mojo
# surface keeps the same validation and Euler equations as a stateless contract
# for accelerator integration.

fn _finite(x: Float64) -> Bool:
    return x == x and x != Float64.INFINITY and x != -Float64.INFINITY

fn nlif_valid(
    v: Float64,
    w: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Bool:
    return (
        _finite(v)
        and _finite(w)
        and _finite(v_rest)
        and _finite(v_crit)
        and _finite(v_threshold)
        and _finite(v_reset)
        and _finite(a)
        and _finite(b)
        and _finite(tau_w)
        and _finite(c_m)
        and _finite(dt)
        and v_rest < v_crit
        and v_crit < v_threshold
        and v_reset < v_threshold
        and a >= 0.0
        and b >= 0.0
        and tau_w > 0.0
        and c_m > 0.0
        and dt > 0.0
        and dt <= tau_w
    )

fn nlif_step_spike(
    v: Float64,
    w: Float64,
    current: Float64,
    v_rest: Float64,
    v_crit: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    a: Float64,
    b: Float64,
    tau_w: Float64,
    c_m: Float64,
    dt: Float64,
) -> Int:
    if not _finite(current):
        return 0
    if not nlif_valid(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt):
        return 0

    var cubic = a * (v - v_rest) * (v - v_crit)
    var dv = (cubic - w + current) / c_m * dt
    var next_v = v + dv
    if next_v >= v_threshold:
        return 1
    return 0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo GPFA EM backend (parity with analysis/spike_stats/gpfa.py)
#
# Build:
#   cd src/sc_neurocore/accel/mojo/kernels
#   mojo build --emit shared-lib -o libgpfa.so gpfa.mojo
#
# Algorithm parity contract: gpfa_em_c runs the GPFA EM loop from a
# caller-supplied deterministic initialisation (PCA, computed once in
# Python) and produces trajectories, parameters and exact marginal
# Gaussian log-likelihoods identical to the NumPy, Rust, Julia and Go
# backends within float64 round-off.
#
# Reference (matches the Python module): Yu, Cunningham, Santhanam, Ryu,
# Shenoy, Sahani (2009) "Gaussian-process factor analysis for
# low-dimensional single-trial analysis of neural population activity",
# J. Neurophysiol. 102:614-635.
#
# Mojo 0.26 FFI rules (per feedback_mojo_026_ffi_pattern):
#   - @export rejects parametric signatures, so every matrix/vector arg
#     is a raw `Int` address (numpy `arr.ctypes.data`) plus size scalars;
#     we reconstruct UnsafePointer[Float64, MutAnyOrigin] inside.
#   - All matrices are flat row-major Float64 (matches the Python + Rust +
#     Julia + Go convention). `tol` is the lone Float64 scalar.

from std.memory import UnsafePointer, alloc
from std.math import log, pi, exp


comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]


# ─── pointer / allocation helpers (mirror lgssm.mojo) ────────────

@always_inline
fn _ptr(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


fn _alloc(n: Int) -> F64Ptr:
    var raw = alloc[Float64](n)
    return F64Ptr(unsafe_from_address=Int(raw))


fn _free(p: F64Ptr):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


@always_inline
fn _fabs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn _zero(p: F64Ptr, n: Int):
    for i in range(n):
        p[i] = 0.0


fn _identity(p: F64Ptr, n: Int):
    _zero(p, n * n)
    for i in range(n):
        p[i * n + i] = 1.0


# ─── dense linear algebra (row-major, parity with the Go backend) ─

# Solve A x = b (A is n×n, b is n×m) via Gauss-Jordan elimination with
# partial pivoting. Inputs are left unmodified; the solution is written
# into the caller-supplied `x` buffer (n×m). A near-singular column is
# skipped (matches the Go reference exactly for bit-comparable parity).
fn _mat_solve(a: F64Ptr, b: F64Ptr, x: F64Ptr, n: Int, m: Int):
    var w = n + m
    var aug = _alloc(n * w)
    for i in range(n):
        for j in range(n):
            aug[i * w + j] = a[i * n + j]
        for j in range(m):
            aug[i * w + n + j] = b[i * m + j]
    for col in range(n):
        var max_row = col
        var max_val = _fabs(aug[col * w + col])
        for row in range(col + 1, n):
            var v = _fabs(aug[row * w + col])
            if v > max_val:
                max_val = v
                max_row = row
        if max_val < 1e-30:
            continue
        if max_row != col:
            for k in range(w):
                var tmp = aug[col * w + k]
                aug[col * w + k] = aug[max_row * w + k]
                aug[max_row * w + k] = tmp
        var pivot = aug[col * w + col]
        for k in range(w):
            aug[col * w + k] /= pivot
        for row in range(n):
            if row == col:
                continue
            var factor = aug[row * w + col]
            for k in range(w):
                aug[row * w + k] -= factor * aug[col * w + k]
    for i in range(n):
        for j in range(m):
            x[i * m + j] = aug[i * w + n + j]
    _free(aug)


# Natural log of the absolute determinant of A (n×n) via LU with partial
# pivoting. A is copied internally and left unmodified. Returns -inf on a
# singular pivot (unreachable for the positive-definite GPFA covariance).
fn _mat_logabsdet(a: F64Ptr, n: Int) -> Float64:
    var m = _alloc(n * n)
    for i in range(n * n):
        m[i] = a[i]
    var log_abs: Float64 = 0.0
    for col in range(n):
        var max_row = col
        var max_val = _fabs(m[col * n + col])
        for row in range(col + 1, n):
            var v = _fabs(m[row * n + col])
            if v > max_val:
                max_val = v
                max_row = row
        if max_val < 1e-300:
            _free(m)
            return log(0.0)
        if max_row != col:
            for k in range(n):
                var tmp = m[col * n + k]
                m[col * n + k] = m[max_row * n + k]
                m[max_row * n + k] = tmp
        var pivot = m[col * n + col]
        log_abs += log(_fabs(pivot))
        for row in range(col + 1, n):
            var factor = m[row * n + col] / pivot
            for k in range(col, n):
                m[row * n + k] -= factor * m[col * n + k]
    _free(m)
    return log_abs


# ─── GPFA EM steps ───────────────────────────────────────────────

# E-step: posterior mean x_out (flat n_latents*n_bins) and the summed
# second moment xx_out (n_latents×n_latents). `k_all` packs the per-latent
# squared-exponential kernels: kernel j occupies offset j*n_bins*n_bins.
fn _e_step(
    y: F64Ptr, c: F64Ptr, d: F64Ptr, r_diag: F64Ptr, k_all: F64Ptr,
    nn: Int, nb: Int, nl: Int,
    x_out: F64Ptr, xx_out: F64Ptr,
):
    var kt = nl * nb

    var r_inv = _alloc(nn)
    for k in range(nn):
        r_inv[k] = 1.0 / (r_diag[k] + 1e-10)

    var ctrinvc = _alloc(nl * nl)
    for i in range(nl):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nn):
                s += c[k * nl + i] * r_inv[k] * c[k * nl + j]
            ctrinvc[i * nl + j] = s

    var ctrinv = _alloc(nl * nn)
    for i in range(nl):
        for k in range(nn):
            ctrinv[i * nn + k] = c[k * nl + i] * r_inv[k]

    var prec = _alloc(kt * kt)
    _zero(prec, kt * kt)
    var eye_bins = _alloc(nb * nb)
    _identity(eye_bins, nb)
    var k_reg = _alloc(nb * nb)
    var k_inv = _alloc(nb * nb)
    for j in range(nl):
        var slj = j * nb
        for i in range(nb * nb):
            k_reg[i] = k_all[j * nb * nb + i]
        for i in range(nb):
            k_reg[i * nb + i] += 1e-6
        _mat_solve(k_reg, eye_bins, k_inv, nb, nb)
        for i in range(nb):
            for jj in range(nb):
                var diag: Float64 = 0.0
                if i == jj:
                    diag = 1.0
                prec[(slj + i) * kt + (slj + jj)] = (
                    k_inv[i * nb + jj] + ctrinvc[j * nl + j] * diag
                )
        for k in range(nl):
            if k != j:
                var slk = k * nb
                for i in range(nb):
                    prec[(slj + i) * kt + (slk + i)] = ctrinvc[j * nl + k]

    var rhs = _alloc(kt)
    for t in range(nb):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nn):
                s += ctrinv[j * nn + k] * (y[k * nb + t] - d[k])
            rhs[j * nb + t] = s
    for i in range(kt):
        prec[i * kt + i] += 1e-8

    _mat_solve(prec, rhs, x_out, kt, 1)

    var eye_kt = _alloc(kt * kt)
    _identity(eye_kt, kt)
    var sigma_post = _alloc(kt * kt)
    _mat_solve(prec, eye_kt, sigma_post, kt, kt)

    _zero(xx_out, nl * nl)
    for t in range(nb):
        for j in range(nl):
            var xj = x_out[j * nb + t]
            for k in range(nl):
                var xk = x_out[k * nb + t]
                xx_out[j * nl + k] += (
                    xj * xk + sigma_post[(j * nb + t) * kt + (k * nb + t)]
                )

    _free(r_inv)
    _free(ctrinvc)
    _free(ctrinv)
    _free(prec)
    _free(eye_bins)
    _free(k_reg)
    _free(k_inv)
    _free(rhs)
    _free(eye_kt)
    _free(sigma_post)


# M-step: update C (n_neurons×n_latents), d (n_neurons) and the noise
# diagonal R (n_neurons). Reads only y, x_post and xx_post, so it may write
# straight into the persistent c/d/r buffers without aliasing hazard.
fn _m_step(
    y: F64Ptr, x_post: F64Ptr, xx_post: F64Ptr,
    nn: Int, nb: Int, nl: Int,
    c_out: F64Ptr, d_out: F64Ptr, r_out: F64Ptr,
):
    for i in range(nn):
        var s: Float64 = 0.0
        for t in range(nb):
            s += y[i * nb + t]
        d_out[i] = s / Float64(nb)

    var yx = _alloc(nn * nl)
    for i in range(nn):
        for j in range(nl):
            var s: Float64 = 0.0
            for t in range(nb):
                s += (y[i * nb + t] - d_out[i]) * x_post[j * nb + t]
            yx[i * nl + j] = s

    var xx_reg = _alloc(nl * nl)
    for i in range(nl * nl):
        xx_reg[i] = xx_post[i]
    for i in range(nl):
        xx_reg[i * nl + i] += 1e-8
    var eye_nl = _alloc(nl * nl)
    _identity(eye_nl, nl)
    var xx_inv = _alloc(nl * nl)
    _mat_solve(xx_reg, eye_nl, xx_inv, nl, nl)

    for i in range(nn):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nl):
                s += yx[i * nl + k] * xx_inv[k * nl + j]
            c_out[i * nl + j] = s

    for i in range(nn):
        var yyt: Float64 = 0.0
        for t in range(nb):
            var v = y[i * nb + t] - d_out[i]
            yyt += v * v
        yyt /= Float64(nb)
        var cxy: Float64 = 0.0
        for j in range(nl):
            for t in range(nb):
                cxy += c_out[i * nl + j] * x_post[j * nb + t] * (y[i * nb + t] - d_out[i])
        cxy /= Float64(nb)
        var rv = yyt - cxy
        if rv < 1e-6:
            rv = 1e-6
        r_out[i] = rv

    _free(yx)
    _free(xx_reg)
    _free(eye_nl)
    _free(xx_inv)


# Exact marginal Gaussian log-likelihood of the GPFA observation model:
# cov = A·K_big·Aᵀ + (I_T ⊗ R) + 1e-8 I, then -0.5(yᵀcov⁻¹y + log|cov| + n logged 2π).
fn _log_likelihood(
    y: F64Ptr, c: F64Ptr, d: F64Ptr, r_diag: F64Ptr, k_all: F64Ptr,
    nn: Int, nb: Int, nl: Int,
) -> Float64:
    var n_obs = nn * nb
    var n_state = nl * nb

    var a = _alloc(n_obs * n_state)
    _zero(a, n_obs * n_state)
    for t in range(nb):
        var row_start = t * nn
        for j in range(nl):
            var col = j * nb + t
            for i in range(nn):
                a[(row_start + i) * n_state + col] = c[i * nl + j]

    var k_big = _alloc(n_state * n_state)
    _zero(k_big, n_state * n_state)
    for j in range(nl):
        for ii in range(nb):
            for jj in range(nb):
                k_big[(j * nb + ii) * n_state + (j * nb + jj)] = k_all[j * nb * nb + ii * nb + jj]

    var ak = _alloc(n_obs * n_state)
    for i in range(n_obs):
        for j in range(n_state):
            var s: Float64 = 0.0
            for k in range(n_state):
                s += a[i * n_state + k] * k_big[k * n_state + j]
            ak[i * n_state + j] = s

    var cov = _alloc(n_obs * n_obs)
    for i in range(n_obs):
        for j in range(n_obs):
            var s: Float64 = 0.0
            for k in range(n_state):
                s += ak[i * n_state + k] * a[j * n_state + k]
            cov[i * n_obs + j] = s
    for t in range(nb):
        for i in range(nn):
            var idx = t * nn + i
            cov[idx * n_obs + idx] += r_diag[i]
    for i in range(n_obs):
        cov[i * n_obs + i] += 1e-8

    var yc = _alloc(n_obs)
    for t in range(nb):
        for i in range(nn):
            yc[t * nn + i] = y[i * nb + t] - d[i]

    var sol = _alloc(n_obs)
    _mat_solve(cov, yc, sol, n_obs, 1)
    var quad: Float64 = 0.0
    for i in range(n_obs):
        quad += yc[i] * sol[i]
    var logdet = _mat_logabsdet(cov, n_obs)
    var result = -0.5 * (quad + logdet + Float64(n_obs) * log(2.0 * pi))

    _free(a)
    _free(k_big)
    _free(ak)
    _free(cov)
    _free(yc)
    _free(sol)
    return result


# ─── exported EM driver ──────────────────────────────────────────

@export
fn gpfa_em_c(
    y_addr: Int, c0_addr: Int, d0_addr: Int, r0_addr: Int, tau_addr: Int,
    n_neurons: Int, n_bins: Int, n_latents: Int, max_iter: Int,
    tol: Float64,
    x_out_addr: Int, params_out_addr: Int, loglik_out_addr: Int,
):
    var y = _ptr(y_addr)
    var c0 = _ptr(c0_addr)
    var d0 = _ptr(d0_addr)
    var r0 = _ptr(r0_addr)
    var tau = _ptr(tau_addr)
    var x_out = _ptr(x_out_addr)
    var params_out = _ptr(params_out_addr)
    var loglik_out = _ptr(loglik_out_addr)

    var nn = n_neurons
    var nb = n_bins
    var nl = n_latents

    # Squared-exponential kernels, packed kernel-j at offset j*nb*nb.
    var k_all = _alloc(nl * nb * nb)
    for j in range(nl):
        var tau_sq = tau[j] * tau[j] + 1e-12
        for i in range(nb):
            for jj in range(nb):
                var diff = Float64(i - jj)
                k_all[j * nb * nb + i * nb + jj] = exp(-0.5 * diff * diff / tau_sq)

    # Working parameters initialised from the caller-supplied PCA init.
    var c = _alloc(nn * nl)
    for i in range(nn * nl):
        c[i] = c0[i]
    var d = _alloc(nn)
    for i in range(nn):
        d[i] = d0[i]
    var r = _alloc(nn)
    for i in range(nn):
        r[i] = r0[i]

    var x_post = _alloc(nl * nb)
    _zero(x_post, nl * nb)
    var xx_post = _alloc(nl * nl)

    var n_iter = 0
    var prev_ll: Float64 = 0.0
    for em_it in range(max_iter):
        _e_step(y, c, d, r, k_all, nn, nb, nl, x_post, xx_post)
        _m_step(y, x_post, xx_post, nn, nb, nl, c, d, r)
        var ll = _log_likelihood(y, c, d, r, k_all, nn, nb, nl)
        loglik_out[1 + em_it] = ll
        n_iter += 1
        if em_it > 0 and _fabs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    loglik_out[0] = Float64(n_iter)
    for i in range(nl * nb):
        x_out[i] = x_post[i]
    for i in range(nn * nl):
        params_out[i] = c[i]
    for i in range(nn):
        params_out[nn * nl + i] = d[i]
    for i in range(nn):
        params_out[nn * nl + nn + i] = r[i]

    _free(k_all)
    _free(c)
    _free(d)
    _free(r)
    _free(x_post)
    _free(xx_post)

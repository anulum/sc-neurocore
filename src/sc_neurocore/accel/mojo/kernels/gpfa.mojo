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
# caller-supplied deterministic initialisation. The linear algebra is
# Cholesky-based and structured: the marginal log-likelihood uses the Woodbury
# identity and the matrix-determinant lemma, so it never forms the dense
# (n_obs × n_obs) covariance and matches the NumPy, Rust, Julia and Go backends
# within float64 round-off.
#
# Reference (matches the Python module): Yu, Cunningham, Santhanam, Ryu, Shenoy,
# Sahani (2009), J. Neurophysiol. 102:614-635.
#
# Mojo 0.26 FFI rules (per feedback_mojo_026_ffi_pattern):
#   - @export rejects parametric signatures, so every array arg is a raw `Int`
#     address (numpy `arr.ctypes.data`) plus size scalars; we reconstruct
#     UnsafePointer[Float64, MutAnyOrigin] inside. `tol` is the lone Float64.
#   - All matrices are flat row-major Float64 (Python/Rust/Julia/Go convention).

from std.memory import UnsafePointer, alloc
from std.math import log, pi, exp, sqrt


comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]


# ─── pointer / allocation helpers (mirror lgssm.mojo) ────────────

@always_inline
def _ptr(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def _alloc(n: Int) -> F64Ptr:
    var raw = alloc[Float64](n)
    return F64Ptr(unsafe_from_address=Int(raw))


def _free(p: F64Ptr):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


def _zero(p: F64Ptr, n: Int):
    for i in range(n):
        p[i] = 0.0


def _identity(p: F64Ptr, n: Int):
    _zero(p, n * n)
    for i in range(n):
        p[i * n + i] = 1.0


# ─── Cholesky-based dense linear algebra (SPD, row-major) ────────

# Lower Cholesky factor L (row-major, upper triangle left at zero) of the
# symmetric positive-definite matrix `a`. GPFA only factors SPD matrices, so
# Cholesky is the stable, ~2x cheaper choice over a general LU elimination.
def _cholesky(a: F64Ptr, n: Int, l: F64Ptr):
    _zero(l, n * n)
    for j in range(n):
        var d = a[j * n + j]
        for k in range(j):
            d -= l[j * n + k] * l[j * n + k]
        var ljj = sqrt(d)
        l[j * n + j] = ljj
        var inv = 1.0 / ljj
        for i in range(j + 1, n):
            var s = a[i * n + j]
            for k in range(j):
                s -= l[i * n + k] * l[j * n + k]
            l[i * n + j] = s * inv


# Solve M X = B for `cols` right-hand sides (B and X row-major n×cols) given the
# lower Cholesky factor L of M, via forward and back substitution. `y_work` is a
# caller-supplied scratch buffer of length n.
def _chol_solve(l: F64Ptr, n: Int, b: F64Ptr, cols: Int, x_out: F64Ptr, y_work: F64Ptr):
    for c in range(cols):
        for i in range(n):
            var s = b[i * cols + c]
            for k in range(i):
                s -= l[i * n + k] * y_work[k]
            y_work[i] = s / l[i * n + i]
        for i_rev in range(n):
            var i = n - 1 - i_rev
            var s = y_work[i]
            for k in range(i + 1, n):
                s -= l[k * n + i] * x_out[k * cols + c]
            x_out[i * cols + c] = s / l[i * n + i]


# log|M| = 2 Σ log L_ii from the Cholesky factor.
def _chol_logdet(l: F64Ptr, n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += log(l[i * n + i])
    return 2.0 * s


# ─── GPFA EM steps ───────────────────────────────────────────────

# Assemble the posterior precision M = blkdiag(K_j⁻¹) + AᵀR⁻¹A (row-major
# n_state×n_state, n_state = nl·nb) into `m_out` and return the GP prior
# log-determinant log|K|. AᵀR⁻¹A adds (CᵀR⁻¹C)[j,k] along the time-diagonal of each
# (j,k) block. Each kernel carries a 1e-6 jitter (the model kernel) and is
# Cholesky-factored once for both its inverse block and its log-determinant.
def _gpfa_precision(
    c: F64Ptr, r_diag: F64Ptr, k_all: F64Ptr,
    nn: Int, nb: Int, nl: Int,
    m_out: F64Ptr,
) -> Float64:
    var n_state = nl * nb
    _zero(m_out, n_state * n_state)

    var r_inv = _alloc(nn)
    for k in range(nn):
        r_inv[k] = 1.0 / r_diag[k]

    var ctrinvc = _alloc(nl * nl)
    for i in range(nl):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nn):
                s += c[k * nl + i] * r_inv[k] * c[k * nl + j]
            ctrinvc[i * nl + j] = s

    var k_reg = _alloc(nb * nb)
    var l_k = _alloc(nb * nb)
    var eye_b = _alloc(nb * nb)
    _identity(eye_b, nb)
    var k_inv = _alloc(nb * nb)
    var y_work = _alloc(nb)
    var logdet_k: Float64 = 0.0
    for j in range(nl):
        for i in range(nb * nb):
            k_reg[i] = k_all[j * nb * nb + i]
        for i in range(nb):
            k_reg[i * nb + i] += 1e-6
        _cholesky(k_reg, nb, l_k)
        logdet_k += _chol_logdet(l_k, nb)
        _chol_solve(l_k, nb, eye_b, nb, k_inv, y_work)
        var slj = j * nb
        for i in range(nb):
            for jj in range(nb):
                m_out[(slj + i) * n_state + (slj + jj)] = k_inv[i * nb + jj]

    for j in range(nl):
        for k in range(nl):
            var v = ctrinvc[j * nl + k]
            for t in range(nb):
                m_out[(j * nb + t) * n_state + (k * nb + t)] += v

    _free(r_inv)
    _free(ctrinvc)
    _free(k_reg)
    _free(l_k)
    _free(eye_b)
    _free(k_inv)
    _free(y_work)
    return logdet_k


# E-step: posterior mean x_out (flat nl·nb) and second moment xx_out (nl×nl). The
# precision M is Cholesky-factored once; the same factor yields the mean
# (M⁻¹ AᵀR⁻¹y) and the covariance (M⁻¹).
def _e_step(
    y: F64Ptr, c: F64Ptr, d: F64Ptr, r_diag: F64Ptr, k_all: F64Ptr,
    nn: Int, nb: Int, nl: Int,
    x_out: F64Ptr, xx_out: F64Ptr,
):
    var n_state = nl * nb
    var r_inv = _alloc(nn)
    for k in range(nn):
        r_inv[k] = 1.0 / r_diag[k]

    var m = _alloc(n_state * n_state)
    var _ld = _gpfa_precision(c, r_diag, k_all, nn, nb, nl, m)

    var rhs = _alloc(n_state)
    for t in range(nb):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nn):
                s += c[k * nl + j] * r_inv[k] * (y[k * nb + t] - d[k])
            rhs[j * nb + t] = s

    var l_m = _alloc(n_state * n_state)
    _cholesky(m, n_state, l_m)
    var y_work = _alloc(n_state)
    _chol_solve(l_m, n_state, rhs, 1, x_out, y_work)

    var eye_s = _alloc(n_state * n_state)
    _identity(eye_s, n_state)
    var sigma = _alloc(n_state * n_state)
    _chol_solve(l_m, n_state, eye_s, n_state, sigma, y_work)

    _zero(xx_out, nl * nl)
    for t in range(nb):
        for j in range(nl):
            var xj = x_out[j * nb + t]
            for k in range(nl):
                var xk = x_out[k * nb + t]
                xx_out[j * nl + k] += xj * xk + sigma[(j * nb + t) * n_state + (k * nb + t)]

    _free(r_inv)
    _free(m)
    _free(rhs)
    _free(l_m)
    _free(y_work)
    _free(eye_s)
    _free(sigma)


# M-step: update C (nn×nl), d (nn) and the noise diagonal R (nn). Reads only y,
# x_post and xx_post, so it writes straight into the persistent c/d/r buffers.
def _m_step(
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
    var l_xx = _alloc(nl * nl)
    _cholesky(xx_reg, nl, l_xx)
    var eye_nl = _alloc(nl * nl)
    _identity(eye_nl, nl)
    var xx_inv = _alloc(nl * nl)
    var y_work = _alloc(nl)
    _chol_solve(l_xx, nl, eye_nl, nl, xx_inv, y_work)

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
    _free(l_xx)
    _free(eye_nl)
    _free(xx_inv)
    _free(y_work)


# Exact marginal Gaussian log-likelihood via the Woodbury identity and the
# matrix-determinant lemma, routed through the n_state×n_state precision M:
#   yᵀ Σ⁻¹ y = yᵀ R⁻¹ y − (AᵀR⁻¹y)ᵀ M⁻¹ (AᵀR⁻¹y)
#   log|Σ|   = log|M| + log|K| + log|R_big|
def _log_likelihood(
    y: F64Ptr, c: F64Ptr, d: F64Ptr, r_diag: F64Ptr, k_all: F64Ptr,
    nn: Int, nb: Int, nl: Int,
) -> Float64:
    var n_obs = nn * nb
    var n_state = nl * nb
    var r_inv = _alloc(nn)
    for k in range(nn):
        r_inv[k] = 1.0 / r_diag[k]

    var m = _alloc(n_state * n_state)
    var logdet_k = _gpfa_precision(c, r_diag, k_all, nn, nb, nl, m)

    var rhs = _alloc(n_state)
    for t in range(nb):
        for j in range(nl):
            var s: Float64 = 0.0
            for k in range(nn):
                s += c[k * nl + j] * r_inv[k] * (y[k * nb + t] - d[k])
            rhs[j * nb + t] = s

    var l_m = _alloc(n_state * n_state)
    _cholesky(m, n_state, l_m)
    var logdet_m = _chol_logdet(l_m, n_state)
    var x_mean = _alloc(n_state)
    var y_work = _alloc(n_state)
    _chol_solve(l_m, n_state, rhs, 1, x_mean, y_work)

    var rhs_x_mean: Float64 = 0.0
    for i in range(n_state):
        rhs_x_mean += rhs[i] * x_mean[i]

    var y_rinv_y: Float64 = 0.0
    for k in range(nn):
        for t in range(nb):
            var v = y[k * nb + t] - d[k]
            y_rinv_y += r_inv[k] * v * v

    var quad = y_rinv_y - rhs_x_mean
    var logdet_r_big: Float64 = 0.0
    for k in range(nn):
        logdet_r_big += log(r_diag[k])
    logdet_r_big *= Float64(nb)
    var logdet_sigma = logdet_m + logdet_k + logdet_r_big
    var result = -0.5 * (quad + logdet_sigma + Float64(n_obs) * log(2.0 * pi))

    _free(r_inv)
    _free(m)
    _free(rhs)
    _free(l_m)
    _free(x_mean)
    _free(y_work)
    return result


# ─── exported EM driver ──────────────────────────────────────────

@export
def gpfa_em_c(
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
        var diff = ll - prev_ll
        if diff < 0.0:
            diff = -diff
        if em_it > 0 and diff < tol:
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

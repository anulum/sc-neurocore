# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo LGSSM Kalman filter (parity with predictive_model.py)
#
# Build:
#   mojo build --emit shared-lib -o liblgssm.so lgssm.mojo
#
# Algorithm parity contract: this kernel produces identical filtered
# means, covariances, and log-likelihood (within float64 round-off)
# to the Python, Rust, Julia, and Go LGSSM implementations under the
# same model parameters and observation sequence.
#
# References (match Python module): Kalman 1960; Bishop 2006 §13.3.1.
#
# Algorithm per timestep t:
#   x_pred = A x_filt + B u
#   P_pred = A P_filt A' + Q
#   e      = y - C x_pred - D u
#   S      = C P_pred C' + R
#   K      = P_pred C' S^{-1}      (computed via Cholesky)
#   x_filt = x_pred + K e
#   P_filt = (I - K C) P_pred (I - K C)' + K R K'   (Joseph form)
#   log_lik += -0.5 (p log 2π + log |S| + e' S^{-1} e)
#
# Mojo C-ABI rules (legacy pointer pattern, compiled with pinned Mojo 1.0.0):
#   - @export rejects parametric signatures, so all matrix/vector
#     args are raw `Int` addresses + size scalars; we reconstruct
#     UnsafePointer[Float64, MutAnyOrigin] inside.
#   - All matrices are flat row-major Float64 (matches the Python +
#     Rust + Julia + Go convention).

from std.memory import UnsafePointer, alloc
from std.math import log, pi, sqrt, nan


# ─── pointer helper ──────────────────────────────────────────────

@always_inline
def _ptr(addr: Int) -> UnsafePointer[Float64, MutAnyOrigin]:
    return UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)


# ─── allocation helper (heap-alloc Float64 buffer of n elements) ──
# `alloc` returns a `MutExternalOrigin` pointer; we re-cast to
# `MutAnyOrigin` via `unsafe_from_address` so the helper signatures
# stay uniform with the input pointers.

def _alloc(n: Int) -> UnsafePointer[Float64, MutAnyOrigin]:
    var raw = alloc[Float64](n)
    return UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(raw))


def _free(p: UnsafePointer[Float64, MutAnyOrigin]):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


# ─── flat row-major matrix ops (rows = r, cols = c) ──────────────

@always_inline
def _at(m: UnsafePointer[Float64, MutAnyOrigin], cols: Int, i: Int, j: Int) -> Float64:
    return m[i * cols + j]


@always_inline
def _set(m: UnsafePointer[Float64, MutAnyOrigin], cols: Int, i: Int, j: Int, v: Float64):
    m[i * cols + j] = v


# C = A · B  where A is (a_rows × a_cols), B is (a_cols × b_cols)
def _matmul(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    a_rows: Int, a_cols: Int, b_cols: Int,
):
    for i in range(a_rows):
        for j in range(b_cols):
            var s: Float64 = 0.0
            for k in range(a_cols):
                s += a[i * a_cols + k] * b[k * b_cols + j]
            c[i * b_cols + j] = s


# C = A · B^T  where A is (a_rows × a_cols), B is (b_rows × a_cols)
# → C shape (a_rows × b_rows).
def _matmul_t(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    c: UnsafePointer[Float64, MutAnyOrigin],
    a_rows: Int, a_cols: Int, b_rows: Int,
):
    for i in range(a_rows):
        for j in range(b_rows):
            var s: Float64 = 0.0
            for k in range(a_cols):
                s += a[i * a_cols + k] * b[j * a_cols + k]
            c[i * b_rows + j] = s


# In-place Cholesky of a symmetric PSD matrix A (n × n), lower
# triangular factor L stored in-place (upper triangle untouched).
# Returns 0 on success, -1 on numerical breakdown (negative pivot).
def _cholesky(a: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Int:
    for j in range(n):
        var diag = a[j * n + j]
        for k in range(j):
            diag -= a[j * n + k] * a[j * n + k]
        if diag <= 0.0:
            return -1
        var l_jj = sqrt(diag)
        a[j * n + j] = l_jj
        var inv_diag = 1.0 / l_jj
        for i in range(j + 1, n):
            var s = a[i * n + j]
            for k in range(j):
                s -= a[i * n + k] * a[j * n + k]
            a[i * n + j] = s * inv_diag
    return 0


# Solve L · x = b in-place into x  (b is consumed → x).
def _trsv_lower(
    l: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
    x: UnsafePointer[Float64, MutAnyOrigin],
):
    for i in range(n):
        var s = x[i]
        for k in range(i):
            s -= l[i * n + k] * x[k]
        x[i] = s / l[i * n + i]


# Solve L^T · x = b in-place into x  (b is consumed → x).
def _trsv_lower_transpose(
    l: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
    x: UnsafePointer[Float64, MutAnyOrigin],
):
    for i_rev in range(n):
        var i = n - 1 - i_rev
        var s = x[i]
        for k in range(i + 1, n):
            s -= l[k * n + i] * x[k]
        x[i] = s / l[i * n + i]


# ─── exported Kalman filter ──────────────────────────────────────

@export
def kalman_filter_c(
    obs_addr: Int, ctl_addr: Int,
    a_addr: Int, b_addr: Int, c_addr: Int, d_addr: Int,
    q_addr: Int, r_addr: Int,
    mu0_addr: Int, sigma0_addr: Int,
    t_len: Int, p_dim: Int, m_dim: Int, d_dim: Int,
    means_out_addr: Int, covs_out_addr: Int,
    pred_means_out_addr: Int, pred_covs_out_addr: Int,
    log_lik_out_addr: Int,
):
    var obs = _ptr(obs_addr)
    var ctl = _ptr(ctl_addr)
    var A = _ptr(a_addr)
    var B = _ptr(b_addr)
    var C = _ptr(c_addr)
    var D = _ptr(d_addr)
    var Q = _ptr(q_addr)
    var R = _ptr(r_addr)
    var mu0 = _ptr(mu0_addr)
    var sigma0 = _ptr(sigma0_addr)
    var means = _ptr(means_out_addr)
    var covs = _ptr(covs_out_addr)
    var pred_means = _ptr(pred_means_out_addr)
    var pred_covs = _ptr(pred_covs_out_addr)
    var log_lik = _ptr(log_lik_out_addr)

    var has_control = m_dim > 0

    # Working buffers (heap-allocated; freed at end)
    var x_pred = _alloc(d_dim)
    var P_pred = _alloc(d_dim * d_dim)
    var x_filt = _alloc(d_dim)
    var P_filt = _alloc(d_dim * d_dim)
    var y_hat = _alloc(p_dim)
    var innov = _alloc(p_dim)
    var S_mat = _alloc(p_dim * p_dim)
    var K_gain = _alloc(d_dim * p_dim)        # K = (P_pred · C^T) · S^{-1}
    var I_minus_KC = _alloc(d_dim * d_dim)
    var tmp_dd = _alloc(d_dim * d_dim)        # scratch (d × d)
    var tmp_dp = _alloc(d_dim * p_dim)        # scratch (d × p)
    var s_inv_innov = _alloc(p_dim)
    var s_inv_col = _alloc(p_dim)              # column of S^{-1} during K solve

    # Initialise: x_pred = mu0, P_pred = sigma0
    for i in range(d_dim):
        x_pred[i] = mu0[i]
    for i in range(d_dim * d_dim):
        P_pred[i] = sigma0[i]

    var log_lik_acc: Float64 = 0.0
    var two_pi_log = log(2.0 * pi)

    for t in range(t_len):
        # Record predicted state
        for i in range(d_dim):
            pred_means[t * d_dim + i] = x_pred[i]
        for i in range(d_dim * d_dim):
            pred_covs[t * d_dim * d_dim + i] = P_pred[i]

        # Innovation: e = y - C x_pred - D u
        # y_hat = C * x_pred
        for i in range(p_dim):
            var s: Float64 = 0.0
            for k in range(d_dim):
                s += C[i * d_dim + k] * x_pred[k]
            y_hat[i] = s
        if has_control:
            for i in range(p_dim):
                var s: Float64 = 0.0
                for k in range(m_dim):
                    s += D[i * m_dim + k] * ctl[t * m_dim + k]
                y_hat[i] += s
        for i in range(p_dim):
            innov[i] = obs[t * p_dim + i] - y_hat[i]

        # S = C * P_pred * C^T + R
        # Step 1: tmp_dp = P_pred * C^T  (d × p)
        _matmul_t(P_pred, C, tmp_dp, d_dim, d_dim, p_dim)
        # Step 2: S_mat = C * tmp_dp + R  (p × p)
        for i in range(p_dim):
            for j in range(p_dim):
                var s: Float64 = 0.0
                for k in range(d_dim):
                    s += C[i * d_dim + k] * tmp_dp[k * p_dim + j]
                S_mat[i * p_dim + j] = s + R[i * p_dim + j]

        # Symmetrise S
        for i in range(p_dim):
            for j in range(i + 1, p_dim):
                var avg = 0.5 * (S_mat[i * p_dim + j] + S_mat[j * p_dim + i])
                S_mat[i * p_dim + j] = avg
                S_mat[j * p_dim + i] = avg

        # Cholesky S = L L^T (in-place into S_mat → lower triangle = L)
        var chol_status = _cholesky(S_mat, p_dim)
        # If Cholesky fails, abort: write NaN into log_lik and exit early
        if chol_status != 0:
            log_lik[0] = nan[DType.float64]()
            _free(x_pred); _free(P_pred); _free(x_filt); _free(P_filt)
            _free(y_hat); _free(innov); _free(S_mat)
            _free(K_gain); _free(I_minus_KC); _free(tmp_dd); _free(tmp_dp)
            _free(s_inv_innov); _free(s_inv_col)
            return

        # log |S| = 2 sum log diag(L)
        var logdet_S: Float64 = 0.0
        for i in range(p_dim):
            logdet_S += 2.0 * log(S_mat[i * p_dim + i])

        # s_inv_innov = S^{-1} * innov via two triangular solves
        for i in range(p_dim):
            s_inv_innov[i] = innov[i]
        _trsv_lower(S_mat, p_dim, s_inv_innov)
        _trsv_lower_transpose(S_mat, p_dim, s_inv_innov)

        # quad form e' S^{-1} e
        var quad: Float64 = 0.0
        for i in range(p_dim):
            quad += innov[i] * s_inv_innov[i]
        log_lik_acc += -0.5 * (Float64(p_dim) * two_pi_log + logdet_S + quad)

        # K = P_pred * C^T * S^{-1}
        # We already have tmp_dp = P_pred * C^T  (d × p).
        # S is symmetric, so solve one transposed row of P_pred * C^T at
        # a time: S * K[i,:]^T = (P_pred * C^T)[i,:]^T.
        for i in range(d_dim):
            for k in range(p_dim):
                s_inv_col[k] = tmp_dp[i * p_dim + k]
            _trsv_lower(S_mat, p_dim, s_inv_col)
            _trsv_lower_transpose(S_mat, p_dim, s_inv_col)
            for k in range(p_dim):
                K_gain[i * p_dim + k] = s_inv_col[k]

        # x_filt = x_pred + K * innov
        for i in range(d_dim):
            var s: Float64 = 0.0
            for k in range(p_dim):
                s += K_gain[i * p_dim + k] * innov[k]
            x_filt[i] = x_pred[i] + s

        # I_minus_KC = I_d - K * C
        for i in range(d_dim):
            for j in range(d_dim):
                var s: Float64 = 0.0
                for k in range(p_dim):
                    s += K_gain[i * p_dim + k] * C[k * d_dim + j]
                var v = -s
                if i == j:
                    v += 1.0
                I_minus_KC[i * d_dim + j] = v

        # P_filt = (I-KC) * P_pred * (I-KC)^T + K * R * K^T
        # Step A: tmp_dd = (I-KC) * P_pred
        _matmul(I_minus_KC, P_pred, tmp_dd, d_dim, d_dim, d_dim)
        # Step B: P_filt = tmp_dd * (I-KC)^T
        _matmul_t(tmp_dd, I_minus_KC, P_filt, d_dim, d_dim, d_dim)
        # Step C: tmp_dp = K * R   (d × p)
        _matmul(K_gain, R, tmp_dp, d_dim, p_dim, p_dim)
        # Step D: P_filt += tmp_dp * K^T  (d × d)
        for i in range(d_dim):
            for j in range(d_dim):
                var s: Float64 = 0.0
                for k in range(p_dim):
                    s += tmp_dp[i * p_dim + k] * K_gain[j * p_dim + k]
                P_filt[i * d_dim + j] += s

        # Record filtered
        for i in range(d_dim):
            means[t * d_dim + i] = x_filt[i]
        for i in range(d_dim * d_dim):
            covs[t * d_dim * d_dim + i] = P_filt[i]

        # Predict next: x_pred = A * x_filt (+ B * u_t)
        for i in range(d_dim):
            var s: Float64 = 0.0
            for k in range(d_dim):
                s += A[i * d_dim + k] * x_filt[k]
            x_pred[i] = s
        if has_control:
            for i in range(d_dim):
                var s: Float64 = 0.0
                for k in range(m_dim):
                    s += B[i * m_dim + k] * ctl[t * m_dim + k]
                x_pred[i] += s
        # P_pred = A * P_filt * A^T + Q
        _matmul(A, P_filt, tmp_dd, d_dim, d_dim, d_dim)
        _matmul_t(tmp_dd, A, P_pred, d_dim, d_dim, d_dim)
        for i in range(d_dim * d_dim):
            P_pred[i] += Q[i]

    log_lik[0] = log_lik_acc

    # Free working buffers
    _free(x_pred); _free(P_pred); _free(x_filt); _free(P_filt)
    _free(y_hat); _free(innov); _free(S_mat)
    _free(K_gain); _free(I_minus_KC); _free(tmp_dd); _free(tmp_dp)
    _free(s_inv_innov); _free(s_inv_col)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Phi* (integrated information) backend
#
# Build:
#   cd src/sc_neurocore/accel/mojo/kernels
#   mojo build --emit shared-lib -o libphi.so phi_estimation.mojo
#
# Geometric integrated information Phi* (Barrett & Seth 2011) under the Gaussian
# assumption. Mutual information is a difference of covariance log-determinants
# taken from Cholesky factors, matching the NumPy, Rust, Julia and Go backends
# within float64 round-off. The lone result is returned through an output address
# (Mojo 0.26 @export: every array arg is a raw int64 address).

from std.memory import UnsafePointer, alloc
from std.math import log, sqrt


comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]


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


# Lower Cholesky factor L (row-major) of the SPD matrix `a`.
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


# log|A| = 2 Σ log Lᵢᵢ for an SPD matrix via Cholesky.
def _logdet_spd(a: F64Ptr, n: Int) -> Float64:
    var l = _alloc(n * n)
    _cholesky(a, n, l)
    var s: Float64 = 0.0
    for i in range(n):
        s += log(l[i * n + i])
    _free(l)
    return 2.0 * s


# Unbiased (ddof=1) row covariance of `data` (n_rows × n_cols) with `eps` jitter on
# the diagonal, written row-major into `cov_out` (n_rows × n_rows).
def _covariance(data: F64Ptr, n_rows: Int, n_cols: Int, eps: Float64, cov_out: F64Ptr):
    var means = _alloc(n_rows)
    for i in range(n_rows):
        var s: Float64 = 0.0
        for t in range(n_cols):
            s += data[i * n_cols + t]
        means[i] = s / Float64(n_cols)
    var denom = Float64(n_cols - 1)
    if denom < 1.0:
        denom = 1.0
    for i in range(n_rows):
        for j in range(i + 1):
            var dot: Float64 = 0.0
            for t in range(n_cols):
                dot += data[i * n_cols + t] * data[j * n_cols + t]
            var v = (dot - Float64(n_cols) * means[i] * means[j]) / denom
            cov_out[i * n_rows + j] = v
            cov_out[j * n_rows + i] = v
        cov_out[i * n_rows + i] += eps
    _free(means)


# Gaussian mutual information MI(X;Y) = 0.5 (log|Cov_X| + log|Cov_Y| - log|Cov_XY|).
def _gaussian_mi(x: F64Ptr, nx: Int, y: F64Ptr, ny: Int, n_cols: Int) -> Float64:
    var eps: Float64 = 1e-10
    var cov_x = _alloc(nx * nx)
    _covariance(x, nx, n_cols, eps, cov_x)
    var cov_y = _alloc(ny * ny)
    _covariance(y, ny, n_cols, eps, cov_y)

    var nxy = nx + ny
    var xy = _alloc(nxy * n_cols)
    for i in range(nx):
        for t in range(n_cols):
            xy[i * n_cols + t] = x[i * n_cols + t]
    for i in range(ny):
        for t in range(n_cols):
            xy[(nx + i) * n_cols + t] = y[i * n_cols + t]
    var cov_xy = _alloc(nxy * nxy)
    _covariance(xy, nxy, n_cols, eps, cov_xy)

    var mi = 0.5 * (_logdet_spd(cov_x, nx) + _logdet_spd(cov_y, ny) - _logdet_spd(cov_xy, nxy))
    _free(cov_x)
    _free(cov_y)
    _free(xy)
    _free(cov_xy)
    if mi < 0.0:
        mi = 0.0
    return mi


@export
def phi_star_c(
    data_addr: Int, n_channels: Int, n_timesteps: Int, tau: Int, out_addr: Int
):
    var data = _ptr(data_addr)
    var out = _ptr(out_addr)
    var n = n_channels
    var t_len = n_timesteps
    if n < 2 or 2 * tau >= t_len:
        out[0] = 0.0
        return

    var tp = t_len - tau
    var past = _alloc(n * tp)
    var future = _alloc(n * tp)
    for i in range(n):
        for c in range(tp):
            past[i * tp + c] = data[i * t_len + c]
            future[i * tp + c] = data[i * t_len + tau + c]

    var mi_whole = _gaussian_mi(past, n, future, n, tp)
    var mi_parts_min: Float64 = 0.0
    var first = True
    for k in range(1, n):
        var mi_a = _gaussian_mi(past, k, future, k, tp)
        var mi_b = _gaussian_mi(past + k * tp, n - k, future + k * tp, n - k, tp)
        var mi_parts = mi_a + mi_b
        if first or mi_parts < mi_parts_min:
            mi_parts_min = mi_parts
            first = False

    var phi = mi_whole - mi_parts_min
    if phi < 0.0:
        phi = 0.0
    out[0] = phi

    _free(past)
    _free(future)

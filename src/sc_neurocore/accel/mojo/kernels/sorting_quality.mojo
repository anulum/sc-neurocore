# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo sorting-quality backend
#
# Build:
#   cd src/sc_neurocore/accel/mojo/kernels
#   mojo build --emit shared-lib -o libsorting_quality.so sorting_quality.mojo
#
# Mahalanobis cluster-quality metrics (Harris et al. 2001; Schmitzer-Torbert et
# al. 2005). The squared Mahalanobis distance is evaluated through the Cholesky
# factor of the regularised cluster covariance with a forward substitution
# (mah = ‖L⁻¹(x-μ)‖²) — the covariance is never inverted — matching the NumPy,
# Rust, Julia and Go backends within float64 round-off. Each scalar result is
# returned through an output address (Mojo 0.26 @export: every array arg is a raw
# int64 address).

from std.memory import UnsafePointer, alloc
from std.math import log, sqrt, exp


comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]


@always_inline
fn _ptr(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


fn _alloc(n: Int) -> F64Ptr:
    var raw = alloc[Float64](n)
    return F64Ptr(unsafe_from_address=Int(raw))


fn _free(p: F64Ptr):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


fn _zero(p: F64Ptr, n: Int):
    for i in range(n):
        p[i] = 0.0


@always_inline
fn _nan() -> Float64:
    # IEEE 0/0 = NaN; the runtime variable prevents constant folding.
    var z: Float64 = 0.0
    return z / z


# Lower Cholesky factor L (row-major) of the SPD matrix `a`.
fn _cholesky(a: F64Ptr, n: Int, l: F64Ptr):
    _zero(l, n * n)
    for j in range(n):
        var d = a[j * n + j]
        for k in range(j):
            d -= l[j * n + k] * l[j * n + k]
        if d <= 0.0:
            d = 1e-300
        var ljj = sqrt(d)
        l[j * n + j] = ljj
        var inv = 1.0 / ljj
        for i in range(j + 1, n):
            var s = a[i * n + j]
            for k in range(j):
                s -= l[i * n + k] * l[j * n + k]
            l[i * n + j] = s * inv


# Unbiased (ddof=1) feature covariance of `data` (n_rows × d) over its rows, with
# `eps` jitter on the diagonal, written row-major into `cov_out` (d × d).
fn _feature_covariance(data: F64Ptr, n_rows: Int, d: Int, eps: Float64, cov_out: F64Ptr):
    var means = _alloc(d)
    for j in range(d):
        means[j] = 0.0
    for i in range(n_rows):
        for j in range(d):
            means[j] += data[i * d + j]
    for j in range(d):
        means[j] /= Float64(n_rows)
    var denom = Float64(n_rows - 1)
    if denom < 1.0:
        denom = 1.0
    _zero(cov_out, d * d)
    for i in range(n_rows):
        for j in range(d):
            var dj = data[i * d + j] - means[j]
            for k in range(j, d):
                cov_out[j * d + k] += dj * (data[i * d + k] - means[k])
    for j in range(d):
        for k in range(j, d):
            cov_out[j * d + k] /= denom
            cov_out[k * d + j] = cov_out[j * d + k]
        cov_out[j * d + j] += eps
    _free(means)


# Squared Mahalanobis distances of each row of `points` (n_pts × d) from the
# cluster mean, via the Cholesky factor of the cluster covariance.
fn _mahalanobis_sq(
    cluster: F64Ptr, n_cluster: Int, points: F64Ptr, n_pts: Int, d: Int, dst: F64Ptr
):
    var mu = _alloc(d)
    for j in range(d):
        mu[j] = 0.0
    for i in range(n_cluster):
        for j in range(d):
            mu[j] += cluster[i * d + j]
    for j in range(d):
        mu[j] /= Float64(n_cluster)

    var cov = _alloc(d * d)
    _feature_covariance(cluster, n_cluster, d, 1e-8, cov)
    var l = _alloc(d * d)
    _cholesky(cov, d, l)

    var z = _alloc(d)
    for p in range(n_pts):
        for j in range(d):
            var s = points[p * d + j] - mu[j]
            for k in range(j):
                s -= l[j * d + k] * z[k]
            z[j] = s / l[j * d + j]
        var m: Float64 = 0.0
        for j in range(d):
            m += z[j] * z[j]
        dst[p] = m

    _free(mu)
    _free(cov)
    _free(l)
    _free(z)


# Insertion sort of `a` (length n) in place — n is the noise count.
fn _sort(a: F64Ptr, n: Int):
    for i in range(1, n):
        var key = a[i]
        var j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


@export
fn isolation_distance_c(
    cluster_addr: Int,
    n_cluster: Int,
    noise_addr: Int,
    n_noise: Int,
    d: Int,
    out_addr: Int,
):
    var dst = _ptr(out_addr)
    if n_cluster < 2 or n_noise < n_cluster or d == 0:
        dst[0] = _nan()
        return
    var cluster = _ptr(cluster_addr)
    var noise = _ptr(noise_addr)

    var mah = _alloc(n_noise)
    _mahalanobis_sq(cluster, n_cluster, noise, n_noise, d, mah)
    _sort(mah, n_noise)
    if n_cluster - 1 < n_noise:
        dst[0] = mah[n_cluster - 1]
    else:
        dst[0] = mah[n_noise - 1]
    _free(mah)


@export
fn l_ratio_c(
    cluster_addr: Int,
    n_cluster: Int,
    noise_addr: Int,
    n_noise: Int,
    d: Int,
    out_addr: Int,
):
    var dst = _ptr(out_addr)
    if n_cluster < 2 or n_noise == 0 or d == 0:
        dst[0] = _nan()
        return
    var cluster = _ptr(cluster_addr)
    var noise = _ptr(noise_addr)

    var mah = _alloc(n_noise)
    _mahalanobis_sq(cluster, n_cluster, noise, n_noise, d, mah)
    var df = Float64(d)
    var s: Float64 = 0.0
    for i in range(n_noise):
        var m = mah[i]
        if m < 1e-10:
            m = 1e-10
        var v = exp(-0.5 * (m - df))
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        s += v
    _free(mah)
    dst[0] = s / Float64(n_cluster)

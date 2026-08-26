# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo dimensionality backend
#
# Build:
#   cd src/sc_neurocore/accel/mojo/kernels
#   mojo build --emit shared-lib -o libdimensionality.so dimensionality.mojo
#
# PCA, demixed PCA (Kobak et al. 2016) and factor analysis (Rubin & Thayer 1982)
# on a pre-binned, mean-centred matrix supplied by the caller. The covariance
# eigendecomposition uses an accurate cyclic-Jacobi solver (no LAPACK is linked)
# with descending eigenvalues and sign-canonicalised eigenvectors; factor
# analysis starts from a deterministic PCA initialisation and solves its
# symmetric positive-definite systems by Cholesky. Matches the NumPy, Rust, Julia
# and Go backends to floating-point round-off. Each result is written through an
# output address (Mojo 0.26 @export: every array arg is a raw int64 address).

from std.memory import UnsafePointer, alloc
from std.math import sqrt, atan2, cos, sin


comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]
comptime IntPtr = UnsafePointer[Int, MutAnyOrigin]


@always_inline
def _ptr(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


@always_inline
def _fabs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _alloc(n: Int) -> F64Ptr:
    var raw = alloc[Float64](n)
    return F64Ptr(unsafe_from_address=Int(raw))


def _free(p: F64Ptr):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


def _alloc_int(n: Int) -> IntPtr:
    var raw = alloc[Int](n)
    return IntPtr(unsafe_from_address=Int(raw))


def _free_int(p: IntPtr):
    var raw = UnsafePointer[Int, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


def _zero(p: F64Ptr, n: Int):
    for i in range(n):
        p[i] = 0.0


# Descending eigenvalues + sign-canonicalised eigenvectors (row-major) of the
# symmetric matrix `a` (n × n) via cyclic Jacobi. Writes into `vals_out` (n) and
# `vecs_out` (n × n, row-major).
def _jacobi_eigen(a_in: F64Ptr, n: Int, vals_out: F64Ptr, vecs_out: F64Ptr):
    var a = _alloc(n * n)
    for i in range(n * n):
        a[i] = a_in[i]
    var v = _alloc(n * n)
    _zero(v, n * n)
    for i in range(n):
        v[i * n + i] = 1.0

    for _sweep in range(100):
        var off: Float64 = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p * n + q] * a[p * n + q]
        if off < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                if _fabs(apq) < 1e-300:
                    continue
                var theta = 0.5 * atan2(2.0 * apq, a[p * n + p] - a[q * n + q])
                var c = cos(theta)
                var s = sin(theta)
                for k in range(n):
                    var akp = a[k * n + p]
                    var akq = a[k * n + q]
                    a[k * n + p] = c * akp + s * akq
                    a[k * n + q] = -s * akp + c * akq
                for k in range(n):
                    var apk = a[p * n + k]
                    var aqk = a[q * n + k]
                    a[p * n + k] = c * apk + s * aqk
                    a[q * n + k] = -s * apk + c * aqk
                for k in range(n):
                    var vkp = v[k * n + p]
                    var vkq = v[k * n + q]
                    v[k * n + p] = c * vkp + s * vkq
                    v[k * n + q] = -s * vkp + c * vkq

    var raw = _alloc(n)
    for i in range(n):
        raw[i] = a[i * n + i]

    # Selection sort of indices by descending eigenvalue.
    var order = _alloc_int(n)
    var used = _alloc_int(n)
    for i in range(n):
        used[i] = 0
    for slot in range(n):
        var best = -1
        var best_val: Float64 = 0.0
        for i in range(n):
            if used[i] == 0 and (best < 0 or raw[i] > best_val):
                best = i
                best_val = raw[i]
        order[slot] = best
        used[best] = 1

    for new_c in range(n):
        var old_c = order[new_c]
        vals_out[new_c] = raw[old_c]
        var piv = 0
        var max_abs: Float64 = 0.0
        for r in range(n):
            var av = _fabs(v[r * n + old_c])
            if av > max_abs:
                max_abs = av
                piv = r
        var sign: Float64 = 1.0
        if v[piv * n + old_c] < 0.0:
            sign = -1.0
        for r in range(n):
            vecs_out[r * n + new_c] = sign * v[r * n + old_c]

    _free(a)
    _free(v)
    _free(raw)
    _free_int(order)
    _free_int(used)


# Lower Cholesky factor L (row-major) of the SPD matrix `a` (n × n).
def _cholesky(a: F64Ptr, n: Int, l: F64Ptr):
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


# Solve A X = B for the Cholesky factor `l` of A (n × n); B row-major n × k →
# X row-major n × k.
def _cholesky_solve(l: F64Ptr, n: Int, b: F64Ptr, k: Int, x: F64Ptr):
    var y = _alloc(n)
    for col in range(k):
        for i in range(n):
            var s = b[i * k + col]
            for j in range(i):
                s -= l[i * n + j] * y[j]
            y[i] = s / l[i * n + i]
        for i in range(n - 1, -1, -1):
            var s = y[i]
            for j in range(i + 1, n):
                s -= l[j * n + i] * x[j * k + col]
            x[i * k + col] = s / l[i * n + i]
    _free(y)


# A⁻¹ for the SPD matrix `a` (n × n) via Cholesky, written into `out` (n × n).
def _spd_inverse(a: F64Ptr, n: Int, out_buf: F64Ptr):
    var l = _alloc(n * n)
    _cholesky(a, n, l)
    var eye = _alloc(n * n)
    _zero(eye, n * n)
    for i in range(n):
        eye[i * n + i] = 1.0
    _cholesky_solve(l, n, eye, n, out_buf)
    _free(l)
    _free(eye)


def _explained(vals: F64Ptr, n: Int, nc: Int, expl_out: F64Ptr):
    var total: Float64 = 0.0
    for i in range(n):
        total += vals[i]
    for c in range(nc):
        if total > 0.0:
            expl_out[c] = vals[c] / total
        else:
            expl_out[c] = vals[c]


def _pca_core(mat: F64Ptr, d: Int, t: Int, nc: Int, proj_out: F64Ptr, expl_out: F64Ptr):
    var denom = Float64(t - 1)
    if denom < 1.0:
        denom = 1.0
    var cov = _alloc(d * d)
    for i in range(d):
        for j in range(i, d):
            var s: Float64 = 0.0
            for k in range(t):
                s += mat[i * t + k] * mat[j * t + k]
            s /= denom
            cov[i * d + j] = s
            cov[j * d + i] = s
    var vals = _alloc(d)
    var vecs = _alloc(d * d)
    _jacobi_eigen(cov, d, vals, vecs)
    _explained(vals, d, nc, expl_out)
    for c in range(nc):
        for tt in range(t):
            var s: Float64 = 0.0
            for i in range(d):
                s += vecs[i * d + c] * mat[i * t + tt]
            proj_out[c * t + tt] = s
    _free(cov)
    _free(vals)
    _free(vecs)


def _demixed_core(
    mean_mat: F64Ptr, n_cond: Int, t: Int, nc: Int, proj_out: F64Ptr, expl_out: F64Ptr
):
    var cov = _alloc(t * t)
    var denom = Float64(n_cond)
    for i in range(t):
        for j in range(i, t):
            var s: Float64 = 0.0
            for c in range(n_cond):
                s += mean_mat[c * t + i] * mean_mat[c * t + j]
            s /= denom
            cov[i * t + j] = s
            cov[j * t + i] = s
    var vals = _alloc(t)
    var vecs = _alloc(t * t)
    _jacobi_eigen(cov, t, vals, vecs)
    _explained(vals, t, nc, expl_out)
    for c in range(n_cond):
        for kk in range(nc):
            var s: Float64 = 0.0
            for j in range(t):
                s += mean_mat[c * t + j] * vecs[j * t + kk]
            proj_out[c * nc + kk] = s
    _free(cov)
    _free(vals)
    _free(vecs)


def _fa_core(mat: F64Ptr, d: Int, t: Int, nf: Int, n_iter: Int, load_out: F64Ptr, psi_out: F64Ptr):
    var tf = Float64(t)
    var cov = _alloc(d * d)
    for i in range(d):
        for j in range(i, d):
            var s: Float64 = 0.0
            for k in range(t):
                s += mat[i * t + k] * mat[j * t + k]
            s /= tf
            cov[i * d + j] = s
            cov[j * d + i] = s
    var vals = _alloc(d)
    var vecs = _alloc(d * d)
    _jacobi_eigen(cov, d, vals, vecs)
    for c in range(nf):
        var scale = sqrt(vals[c]) if vals[c] > 0.0 else 0.0
        for i in range(d):
            load_out[i * nf + c] = vecs[i * d + c] * scale
    for i in range(d):
        psi_out[i] = cov[i * d + i]

    var psi_inv = _alloc(d)
    var m = _alloc(nf * nf)
    var m_inv = _alloc(nf * nf)
    var beta = _alloc(nf * d)
    var ez = _alloc(nf * t)
    var ezzt = _alloc(nf * nf)
    var mat_ez_t = _alloc(d * nf)
    var rhs = _alloc(nf * d)
    var lez = _alloc(nf * nf)
    var solved = _alloc(nf * d)

    for _iter in range(n_iter):
        for i in range(d):
            psi_inv[i] = 1.0 / (psi_out[i] + 1e-10)
        for a in range(nf):
            for b in range(nf):
                var s: Float64 = 0.0
                for i in range(d):
                    s += load_out[i * nf + a] * psi_inv[i] * load_out[i * nf + b]
                if a == b:
                    s += 1.0
                m[a * nf + b] = s
        _spd_inverse(m, nf, m_inv)
        for a in range(nf):
            for i in range(d):
                var s: Float64 = 0.0
                for kk in range(nf):
                    s += m_inv[a * nf + kk] * load_out[i * nf + kk] * psi_inv[i]
                beta[a * d + i] = s
        for a in range(nf):
            for tt in range(t):
                var s: Float64 = 0.0
                for i in range(d):
                    s += beta[a * d + i] * mat[i * t + tt]
                ez[a * t + tt] = s
        for a in range(nf):
            for b in range(nf):
                var s: Float64 = 0.0
                for tt in range(t):
                    s += ez[a * t + tt] * ez[b * t + tt]
                ezzt[a * nf + b] = Float64(nf) * m_inv[a * nf + b] + s / tf
        for i in range(d):
            for a in range(nf):
                var s: Float64 = 0.0
                for tt in range(t):
                    s += mat[i * t + tt] * ez[a * t + tt]
                mat_ez_t[i * nf + a] = s / tf
        for a in range(nf):
            for i in range(d):
                rhs[a * d + i] = mat_ez_t[i * nf + a]
        _cholesky(ezzt, nf, lez)
        _cholesky_solve(lez, nf, rhs, d, solved)
        for i in range(d):
            for a in range(nf):
                load_out[i * nf + a] = solved[a * d + i]
        for i in range(d):
            var s: Float64 = 0.0
            for tt in range(t):
                var le: Float64 = 0.0
                for a in range(nf):
                    le += load_out[i * nf + a] * ez[a * t + tt]
                s += le * mat[i * t + tt]
            var p = cov[i * d + i] - s / tf
            psi_out[i] = p if p > 1e-6 else 1e-6

    _free(cov)
    _free(vals)
    _free(vecs)
    _free(psi_inv)
    _free(m)
    _free(m_inv)
    _free(beta)
    _free(ez)
    _free(ezzt)
    _free(mat_ez_t)
    _free(rhs)
    _free(lez)
    _free(solved)


@export
def pca_from_matrix_c(
    mat_addr: Int, d: Int, t: Int, nc: Int, proj_addr: Int, expl_addr: Int
):
    _pca_core(_ptr(mat_addr), d, t, nc, _ptr(proj_addr), _ptr(expl_addr))


@export
def demixed_from_matrix_c(
    mat_addr: Int, n_cond: Int, t: Int, nc: Int, proj_addr: Int, expl_addr: Int
):
    _demixed_core(_ptr(mat_addr), n_cond, t, nc, _ptr(proj_addr), _ptr(expl_addr))


@export
def factor_analysis_c(
    mat_addr: Int, d: Int, t: Int, nf: Int, load_addr: Int, psi_addr: Int, n_iter: Int
):
    _fa_core(_ptr(mat_addr), d, t, nf, n_iter, _ptr(load_addr), _ptr(psi_addr))

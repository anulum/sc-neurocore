// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go dimensionality backend (parity with
// analysis/spike_stats/dimensionality.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libdimensionality.so`) that the Python
// dispatcher loads via ctypes. pca_from_matrix_c, demixed_from_matrix_c and
// factor_analysis_c operate on a pre-binned, mean-centred matrix supplied by the
// caller, so every backend shares an identical input. The covariance
// eigendecomposition uses an accurate cyclic-Jacobi solver (no LAPACK is linked
// into the Go runtime) with descending eigenvalues and sign-canonicalised
// eigenvectors; factor analysis starts from a deterministic PCA initialisation
// and solves its symmetric positive-definite systems by Cholesky. Results match
// the NumPy, Rust, Julia and Mojo backends to floating-point round-off.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"sort"
	"unsafe"
)

// jacobiEigen returns the descending eigenvalues and sign-canonicalised
// eigenvectors (row-major, vecs[r*n+c]) of the symmetric matrix a (n x n).
func jacobiEigen(aIn []float64, n int) ([]float64, []float64) {
	a := make([]float64, n*n)
	copy(a, aIn)
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1.0
	}
	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += a[p*n+q] * a[p*n+q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				theta := 0.5 * math.Atan2(2*apq, a[p*n+p]-a[q*n+q])
				c := math.Cos(theta)
				s := math.Sin(theta)
				for k := 0; k < n; k++ {
					akp := a[k*n+p]
					akq := a[k*n+q]
					a[k*n+p] = c*akp + s*akq
					a[k*n+q] = -s*akp + c*akq
				}
				for k := 0; k < n; k++ {
					apk := a[p*n+k]
					aqk := a[q*n+k]
					a[p*n+k] = c*apk + s*aqk
					a[q*n+k] = -s*apk + c*aqk
				}
				for k := 0; k < n; k++ {
					vkp := v[k*n+p]
					vkq := v[k*n+q]
					v[k*n+p] = c*vkp + s*vkq
					v[k*n+q] = -s*vkp + c*vkq
				}
			}
		}
	}
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = a[i*n+i]
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return vals[idx[i]] > vals[idx[j]] })

	sortedVals := make([]float64, n)
	vecs := make([]float64, n*n)
	for newc, oldc := range idx {
		sortedVals[newc] = vals[oldc]
		piv := 0
		maxAbs := 0.0
		for r := 0; r < n; r++ {
			if math.Abs(v[r*n+oldc]) > maxAbs {
				maxAbs = math.Abs(v[r*n+oldc])
				piv = r
			}
		}
		sign := 1.0
		if v[piv*n+oldc] < 0 {
			sign = -1.0
		}
		for r := 0; r < n; r++ {
			vecs[r*n+newc] = sign * v[r*n+oldc]
		}
	}
	return sortedVals, vecs
}

// cholesky returns the lower factor L (row-major) of the SPD matrix a (n x n).
func cholesky(a []float64, n int) []float64 {
	l := make([]float64, n*n)
	for j := 0; j < n; j++ {
		d := a[j*n+j]
		for k := 0; k < j; k++ {
			d -= l[j*n+k] * l[j*n+k]
		}
		if d <= 0 {
			d = 1e-300
		}
		ljj := math.Sqrt(d)
		l[j*n+j] = ljj
		inv := 1.0 / ljj
		for i := j + 1; i < n; i++ {
			s := a[i*n+j]
			for k := 0; k < j; k++ {
				s -= l[i*n+k] * l[j*n+k]
			}
			l[i*n+j] = s * inv
		}
	}
	return l
}

// choleskySolve solves A X = B for the Cholesky factor l of A (n x n), B row-major
// n x k → X row-major n x k.
func choleskySolve(l []float64, n int, b []float64, k int) []float64 {
	x := make([]float64, n*k)
	y := make([]float64, n)
	for col := 0; col < k; col++ {
		for i := 0; i < n; i++ {
			s := b[i*k+col]
			for j := 0; j < i; j++ {
				s -= l[i*n+j] * y[j]
			}
			y[i] = s / l[i*n+i]
		}
		for i := n - 1; i >= 0; i-- {
			s := y[i]
			for j := i + 1; j < n; j++ {
				s -= l[j*n+i] * x[j*k+col]
			}
			x[i*k+col] = s / l[i*n+i]
		}
	}
	return x
}

// spdInverse returns A⁻¹ for the SPD matrix a (n x n) via Cholesky.
func spdInverse(a []float64, n int) []float64 {
	l := cholesky(a, n)
	eye := make([]float64, n*n)
	for i := 0; i < n; i++ {
		eye[i*n+i] = 1.0
	}
	return choleskySolve(l, n, eye, n)
}

func explainedRatio(vals []float64, nc int) []float64 {
	total := 0.0
	for _, v := range vals {
		total += v
	}
	out := make([]float64, nc)
	for i := 0; i < nc; i++ {
		if total > 0 {
			out[i] = vals[i] / total
		} else {
			out[i] = vals[i]
		}
	}
	return out
}

// pcaFromCentered: PCA of a centred d x t matrix → (projected nc x t, explained nc).
func pcaFromCentered(mat []float64, d, t, nComp int) ([]float64, []float64) {
	denom := math.Max(float64(t-1), 1.0)
	cov := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			s := 0.0
			for k := 0; k < t; k++ {
				s += mat[i*t+k] * mat[j*t+k]
			}
			s /= denom
			cov[i*d+j] = s
			cov[j*d+i] = s
		}
	}
	vals, vecs := jacobiEigen(cov, d)
	nc := nComp
	if nc > d {
		nc = d
	}
	proj := make([]float64, nc*t)
	for c := 0; c < nc; c++ {
		for tt := 0; tt < t; tt++ {
			s := 0.0
			for i := 0; i < d; i++ {
				s += vecs[i*d+c] * mat[i*t+tt]
			}
			proj[c*t+tt] = s
		}
	}
	return proj, explainedRatio(vals, nc)
}

// demixedFromCentered: demixed PCA of a centred n_cond x t matrix.
func demixedFromCentered(meanMat []float64, nCond, t, nComp int) ([]float64, []float64) {
	cov := make([]float64, t*t)
	denom := float64(nCond)
	for i := 0; i < t; i++ {
		for j := i; j < t; j++ {
			s := 0.0
			for c := 0; c < nCond; c++ {
				s += meanMat[c*t+i] * meanMat[c*t+j]
			}
			s /= denom
			cov[i*t+j] = s
			cov[j*t+i] = s
		}
	}
	vals, vecs := jacobiEigen(cov, t)
	nc := nComp
	if nc > t {
		nc = t
	}
	proj := make([]float64, nCond*nc)
	for c := 0; c < nCond; c++ {
		for k := 0; k < nc; k++ {
			s := 0.0
			for j := 0; j < t; j++ {
				s += meanMat[c*t+j] * vecs[j*t+k]
			}
			proj[c*nc+k] = s
		}
	}
	return proj, explainedRatio(vals, nc)
}

// faFromCentered: factor-analysis EM of a centred d x t matrix (deterministic init).
func faFromCentered(mat []float64, d, t, nFactors, nIter int) ([]float64, []float64) {
	tf := float64(t)
	cov := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			s := 0.0
			for k := 0; k < t; k++ {
				s += mat[i*t+k] * mat[j*t+k]
			}
			s /= tf
			cov[i*d+j] = s
			cov[j*d+i] = s
		}
	}
	nf := nFactors
	if nf > d {
		nf = d
	}
	vals, vecs := jacobiEigen(cov, d)
	loadings := make([]float64, d*nf)
	for c := 0; c < nf; c++ {
		scale := math.Sqrt(math.Max(vals[c], 0.0))
		for i := 0; i < d; i++ {
			loadings[i*nf+c] = vecs[i*d+c] * scale
		}
	}
	psi := make([]float64, d)
	for i := 0; i < d; i++ {
		psi[i] = cov[i*d+i]
	}

	for iter := 0; iter < nIter; iter++ {
		psiInv := make([]float64, d)
		for i := 0; i < d; i++ {
			psiInv[i] = 1.0 / (psi[i] + 1e-10)
		}
		m := make([]float64, nf*nf)
		for a := 0; a < nf; a++ {
			for b := 0; b < nf; b++ {
				s := 0.0
				for i := 0; i < d; i++ {
					s += loadings[i*nf+a] * psiInv[i] * loadings[i*nf+b]
				}
				if a == b {
					s += 1.0
				}
				m[a*nf+b] = s
			}
		}
		mInv := spdInverse(m, nf)
		beta := make([]float64, nf*d)
		for a := 0; a < nf; a++ {
			for i := 0; i < d; i++ {
				s := 0.0
				for kk := 0; kk < nf; kk++ {
					s += mInv[a*nf+kk] * loadings[i*nf+kk] * psiInv[i]
				}
				beta[a*d+i] = s
			}
		}
		ez := make([]float64, nf*t)
		for a := 0; a < nf; a++ {
			for tt := 0; tt < t; tt++ {
				s := 0.0
				for i := 0; i < d; i++ {
					s += beta[a*d+i] * mat[i*t+tt]
				}
				ez[a*t+tt] = s
			}
		}
		ezzt := make([]float64, nf*nf)
		for a := 0; a < nf; a++ {
			for b := 0; b < nf; b++ {
				s := 0.0
				for tt := 0; tt < t; tt++ {
					s += ez[a*t+tt] * ez[b*t+tt]
				}
				ezzt[a*nf+b] = float64(nf)*mInv[a*nf+b] + s/tf
			}
		}
		matEzT := make([]float64, d*nf)
		for i := 0; i < d; i++ {
			for a := 0; a < nf; a++ {
				s := 0.0
				for tt := 0; tt < t; tt++ {
					s += mat[i*t+tt] * ez[a*t+tt]
				}
				matEzT[i*nf+a] = s / tf
			}
		}
		rhs := make([]float64, nf*d)
		for a := 0; a < nf; a++ {
			for i := 0; i < d; i++ {
				rhs[a*d+i] = matEzT[i*nf+a]
			}
		}
		lez := cholesky(ezzt, nf)
		solved := choleskySolve(lez, nf, rhs, d)
		for i := 0; i < d; i++ {
			for a := 0; a < nf; a++ {
				loadings[i*nf+a] = solved[a*d+i]
			}
		}
		for i := 0; i < d; i++ {
			s := 0.0
			for tt := 0; tt < t; tt++ {
				le := 0.0
				for a := 0; a < nf; a++ {
					le += loadings[i*nf+a] * ez[a*t+tt]
				}
				s += le * mat[i*t+tt]
			}
			psi[i] = math.Max(cov[i*d+i]-s/tf, 1e-6)
		}
	}
	return loadings, psi
}

//export pca_from_matrix_c
func pca_from_matrix_c(matPtr *C.double, d, t, nc C.int, projOut, explOut *C.double) {
	dd, tt, ncc := int(d), int(t), int(nc)
	mat := unsafe.Slice((*float64)(unsafe.Pointer(matPtr)), dd*tt)
	proj, expl := pcaFromCentered(mat, dd, tt, ncc)
	copy(unsafe.Slice((*float64)(unsafe.Pointer(projOut)), ncc*tt), proj)
	copy(unsafe.Slice((*float64)(unsafe.Pointer(explOut)), ncc), expl)
}

//export demixed_from_matrix_c
func demixed_from_matrix_c(matPtr *C.double, nCond, t, nc C.int, projOut, explOut *C.double) {
	nc2, tt, ncc := int(nCond), int(t), int(nc)
	mat := unsafe.Slice((*float64)(unsafe.Pointer(matPtr)), nc2*tt)
	proj, expl := demixedFromCentered(mat, nc2, tt, ncc)
	copy(unsafe.Slice((*float64)(unsafe.Pointer(projOut)), nc2*ncc), proj)
	copy(unsafe.Slice((*float64)(unsafe.Pointer(explOut)), ncc), expl)
}

//export factor_analysis_c
func factor_analysis_c(matPtr *C.double, d, t, nf C.int, loadOut, psiOut *C.double, nIter C.int) {
	dd, tt, nff := int(d), int(t), int(nf)
	mat := unsafe.Slice((*float64)(unsafe.Pointer(matPtr)), dd*tt)
	loadings, psi := faFromCentered(mat, dd, tt, nff, int(nIter))
	copy(unsafe.Slice((*float64)(unsafe.Pointer(loadOut)), dd*nff), loadings)
	copy(unsafe.Slice((*float64)(unsafe.Pointer(psiOut)), dd), psi)
}

func main() {} // required for c-shared

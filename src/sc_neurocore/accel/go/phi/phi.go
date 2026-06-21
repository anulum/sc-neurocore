// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Phi* backend (parity with analysis/phi_estimation.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libphi.so`) that the Python dispatcher loads
// via ctypes. phi_star_c computes the geometric integrated information Phi*
// (Barrett & Seth 2011) under the Gaussian assumption: mutual information is a
// difference of covariance log-determinants taken from Cholesky factors, matching
// the NumPy, Rust, Julia and Mojo backends within float64 round-off.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// cholesky returns the lower Cholesky factor L (row-major) of the SPD matrix a,
// and false on a non-positive pivot.
func cholesky(a []float64, n int) ([]float64, bool) {
	l := make([]float64, n*n)
	for j := 0; j < n; j++ {
		d := a[j*n+j]
		for k := 0; k < j; k++ {
			d -= l[j*n+k] * l[j*n+k]
		}
		if d <= 0 {
			return nil, false
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
	return l, true
}

// logdetSpd returns log|A| = 2 sum log L_ii for an SPD matrix via Cholesky.
func logdetSpd(cov []float64, n int) float64 {
	l, ok := cholesky(cov, n)
	if !ok {
		panic("logdetSpd: covariance is not positive definite")
	}
	s := 0.0
	for i := 0; i < n; i++ {
		s += math.Log(l[i*n+i])
	}
	return 2.0 * s
}

// covariance returns the unbiased (ddof=1) row covariance of data (nRows x nCols)
// with eps jitter on the diagonal, row-major (nRows x nRows).
func covariance(data []float64, nRows, nCols int, eps float64) []float64 {
	means := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		s := 0.0
		for t := 0; t < nCols; t++ {
			s += data[i*nCols+t]
		}
		means[i] = s / float64(nCols)
	}
	denom := math.Max(float64(nCols-1), 1.0)
	cov := make([]float64, nRows*nRows)
	for i := 0; i < nRows; i++ {
		for j := 0; j <= i; j++ {
			dot := 0.0
			for t := 0; t < nCols; t++ {
				dot += data[i*nCols+t] * data[j*nCols+t]
			}
			v := (dot - float64(nCols)*means[i]*means[j]) / denom
			cov[i*nRows+j] = v
			cov[j*nRows+i] = v
		}
		cov[i*nRows+i] += eps
	}
	return cov
}

// gaussianMI returns MI(X;Y) = 0.5 (log|Cov_X| + log|Cov_Y| - log|Cov_XY|).
func gaussianMI(x []float64, nx int, y []float64, ny int, nCols int) float64 {
	eps := 1e-10
	covX := covariance(x, nx, nCols, eps)
	covY := covariance(y, ny, nCols, eps)

	nxy := nx + ny
	xy := make([]float64, nxy*nCols)
	for i := 0; i < nx; i++ {
		copy(xy[i*nCols:(i+1)*nCols], x[i*nCols:(i+1)*nCols])
	}
	for i := 0; i < ny; i++ {
		copy(xy[(nx+i)*nCols:(nx+i+1)*nCols], y[i*nCols:(i+1)*nCols])
	}
	covXY := covariance(xy, nxy, nCols, eps)

	mi := 0.5 * (logdetSpd(covX, nx) + logdetSpd(covY, ny) - logdetSpd(covXY, nxy))
	return math.Max(0.0, mi)
}

// phiStar computes geometric Phi* for a flat row-major (n x tLen) matrix.
func phiStar(data []float64, n, tLen, tau int) float64 {
	if n < 2 || 2*tau >= tLen {
		return 0.0
	}
	tp := tLen - tau
	past := make([]float64, n*tp)
	future := make([]float64, n*tp)
	for i := 0; i < n; i++ {
		for c := 0; c < tp; c++ {
			past[i*tp+c] = data[i*tLen+c]
			future[i*tp+c] = data[i*tLen+tau+c]
		}
	}

	miWhole := gaussianMI(past, n, future, n, tp)
	miPartsMin := math.Inf(1)
	for k := 1; k < n; k++ {
		miA := gaussianMI(past[:k*tp], k, future[:k*tp], k, tp)
		miB := gaussianMI(past[k*tp:], n-k, future[k*tp:], n-k, tp)
		if mp := miA + miB; mp < miPartsMin {
			miPartsMin = mp
		}
	}
	return math.Max(0.0, miWhole-miPartsMin)
}

//export phi_star_c
func phi_star_c(dataPtr *C.double, nChannels, nTimesteps, tau C.int) C.double {
	n := int(nChannels)
	tLen := int(nTimesteps)
	tu := int(tau)
	if n == 0 || tLen == 0 {
		return C.double(0.0)
	}
	data := unsafe.Slice((*float64)(unsafe.Pointer(dataPtr)), n*tLen)
	return C.double(phiStar(data, n, tLen, tu))
}

func main() {} // required for c-shared

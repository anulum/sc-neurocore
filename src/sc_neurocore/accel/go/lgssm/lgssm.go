// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go LGSSM Kalman filter (parity with predictive_model.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o liblgssm.so`) that the Python
// dispatcher loads via ctypes.
//
// Algorithm parity contract: the function `kalman_filter` produces
// identical filtered means, covariances, and log-likelihood (within
// float64 round-off) to the Python and Rust LGSSM implementations
// under the same model parameters and observation sequence.
//
// References (match Python module):
//   Kalman 1960; Bishop 2006 §13.3.1.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// kalmanFilter runs the forward Kalman filter on a sequence.
//
// All matrices passed as flat row-major slices. Returns nothing —
// fills the output buffers in-place. The caller (Python ctypes
// wrapper) allocates the output buffers with the correct shapes.
//
// Inputs (caller-allocated, read-only):
//   obsFlat:     T*p elements, observations[t, p]
//   ctlFlat:     T*m elements, controls[t, m]
//   aFlat:       d*d, A
//   bFlat:       d*m, B
//   cFlat:       p*d, C
//   dFlatMat:    p*m, D
//   qFlat:       d*d, Q
//   rFlat:       p*p, R
//   mu0:         d,   mu_0
//   sigma0Flat:  d*d, Sigma_0
//   tLen:        T
//   pDim:        p
//   mDim:        m
//   dDim:        d
//
// Outputs (caller-allocated, written):
//   meansOut:        T*d
//   covsOut:         T*d*d
//   predMeansOut:    T*d
//   predCovsOut:     T*d*d
//   logLikOut:       1
func kalmanFilter(
	obsFlat, ctlFlat []float64,
	aFlat, bFlat, cFlat, dFlatMat, qFlat, rFlat []float64,
	mu0, sigma0Flat []float64,
	tLen, pDim, mDim, dDim int,
	meansOut, covsOut, predMeansOut, predCovsOut []float64,
	logLikOut []float64,
) {
	hasControl := mDim > 0

	// Helpers to view flat slices as row-major matrices
	idx2 := func(rows int) func(i, j int) int {
		_ = rows
		return func(i, j int) int { return i*0 + j } // placeholder
	}
	_ = idx2

	get := func(buf []float64, rows, cols, i, j int) float64 {
		_ = rows
		return buf[i*cols+j]
	}
	set := func(buf []float64, rows, cols, i, j int, v float64) {
		_ = rows
		buf[i*cols+j] = v
	}

	// State vectors
	xPred := make([]float64, dDim)
	pPred := make([]float64, dDim*dDim)
	copy(xPred, mu0)
	copy(pPred, sigma0Flat)

	twoPiLog := math.Log(2 * math.Pi)
	logLik := 0.0

	// Workspace buffers
	yHat := make([]float64, pDim)
	innov := make([]float64, pDim)
	cPpred := make([]float64, pDim*dDim) // C @ P_pred, shape (p, d)
	sMat := make([]float64, pDim*pDim)
	sChol := make([]float64, pDim*pDim)
	sInvInnov := make([]float64, pDim)
	kGain := make([]float64, dDim*pDim) // K = P_pred C^T S^{-1}
	xFilt := make([]float64, dDim)
	pFilt := make([]float64, dDim*dDim)
	imkc := make([]float64, dDim*dDim)
	tmpDD := make([]float64, dDim*dDim)
	tmpDD2 := make([]float64, dDim*dDim)
	tmpDP := make([]float64, dDim*pDim)

	for t := 0; t < tLen; t++ {
		// Record predicted
		for i := 0; i < dDim; i++ {
			set(predMeansOut, tLen, dDim, t, i, xPred[i])
		}
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				predCovsOut[t*dDim*dDim+i*dDim+j] = get(pPred, dDim, dDim, i, j)
			}
		}

		// Innovation: e = y - C x_pred - D u
		for i := 0; i < pDim; i++ {
			s := 0.0
			for j := 0; j < dDim; j++ {
				s += get(cFlat, pDim, dDim, i, j) * xPred[j]
			}
			yHat[i] = s
		}
		if hasControl {
			for i := 0; i < pDim; i++ {
				s := 0.0
				for j := 0; j < mDim; j++ {
					s += get(dFlatMat, pDim, mDim, i, j) * get(ctlFlat, tLen, mDim, t, j)
				}
				yHat[i] += s
			}
		}
		for i := 0; i < pDim; i++ {
			innov[i] = get(obsFlat, tLen, pDim, t, i) - yHat[i]
		}

		// C @ P_pred → cPpred (p, d)
		for i := 0; i < pDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += get(cFlat, pDim, dDim, i, k) * get(pPred, dDim, dDim, k, j)
				}
				cPpred[i*dDim+j] = s
			}
		}
		// S = C P_pred C^T + R
		for i := 0; i < pDim; i++ {
			for j := 0; j < pDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += cPpred[i*dDim+k] * get(cFlat, pDim, dDim, j, k)
				}
				sMat[i*pDim+j] = s + get(rFlat, pDim, pDim, i, j)
			}
		}

		// Cholesky of S
		choleskyInPlace(sMat, sChol, pDim)
		// log-determinant via 2 sum log diag(L)
		logDet := 0.0
		for i := 0; i < pDim; i++ {
			logDet += math.Log(sChol[i*pDim+i])
		}
		logDet *= 2.0

		// Solve S z = innov via Cholesky (forward + backward)
		choleskySolve(sChol, innov, sInvInnov, pDim)

		// Quadratic form
		quadForm := 0.0
		for i := 0; i < pDim; i++ {
			quadForm += innov[i] * sInvInnov[i]
		}
		logLik += -0.5 * (float64(pDim)*twoPiLog + logDet + quadForm)

		// Kalman gain K = P_pred C^T S^{-1}
		// First: tmpDP = P_pred C^T (d, p)
		for i := 0; i < dDim; i++ {
			for j := 0; j < pDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += get(pPred, dDim, dDim, i, k) * get(cFlat, pDim, dDim, j, k)
				}
				tmpDP[i*pDim+j] = s
			}
		}
		// K = tmpDP @ S^{-1}: solve column by column
		for col := 0; col < pDim; col++ {
			rhs := make([]float64, pDim)
			out := make([]float64, pDim)
			for i := 0; i < pDim; i++ {
				rhs[i] = 0.0
			}
			rhs[col] = 1.0
			choleskySolve(sChol, rhs, out, pDim)
			for i := 0; i < dDim; i++ {
				s := 0.0
				for k := 0; k < pDim; k++ {
					s += tmpDP[i*pDim+k] * out[k]
				}
				kGain[i*pDim+col] = s
			}
		}

		// x_filt = x_pred + K e
		for i := 0; i < dDim; i++ {
			s := xPred[i]
			for j := 0; j < pDim; j++ {
				s += kGain[i*pDim+j] * innov[j]
			}
			xFilt[i] = s
		}

		// I_minus_KC = I - K C  (d, d)
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < pDim; k++ {
					s += kGain[i*pDim+k] * get(cFlat, pDim, dDim, k, j)
				}
				if i == j {
					imkc[i*dDim+j] = 1.0 - s
				} else {
					imkc[i*dDim+j] = -s
				}
			}
		}
		// tmpDD = imkc @ P_pred
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += imkc[i*dDim+k] * get(pPred, dDim, dDim, k, j)
				}
				tmpDD[i*dDim+j] = s
			}
		}
		// tmpDD2 = tmpDD @ imkc^T
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += tmpDD[i*dDim+k] * imkc[j*dDim+k]
				}
				tmpDD2[i*dDim+j] = s
			}
		}
		// pFilt = tmpDD2 + K R K^T
		// First: tmpDP2 = K R (d, p)
		tmpDP2 := make([]float64, dDim*pDim)
		for i := 0; i < dDim; i++ {
			for j := 0; j < pDim; j++ {
				s := 0.0
				for k := 0; k < pDim; k++ {
					s += kGain[i*pDim+k] * get(rFlat, pDim, pDim, k, j)
				}
				tmpDP2[i*pDim+j] = s
			}
		}
		// pFilt = tmpDD2 + tmpDP2 @ K^T
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < pDim; k++ {
					s += tmpDP2[i*pDim+k] * kGain[j*pDim+k]
				}
				pFilt[i*dDim+j] = tmpDD2[i*dDim+j] + s
			}
		}

		// Record filtered state
		for i := 0; i < dDim; i++ {
			set(meansOut, tLen, dDim, t, i, xFilt[i])
		}
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				covsOut[t*dDim*dDim+i*dDim+j] = pFilt[i*dDim+j]
			}
		}

		// Predict next: x_pred = A x_filt + B u
		for i := 0; i < dDim; i++ {
			s := 0.0
			for j := 0; j < dDim; j++ {
				s += get(aFlat, dDim, dDim, i, j) * xFilt[j]
			}
			xPred[i] = s
		}
		if hasControl {
			for i := 0; i < dDim; i++ {
				s := 0.0
				for j := 0; j < mDim; j++ {
					s += get(bFlat, dDim, mDim, i, j) * get(ctlFlat, tLen, mDim, t, j)
				}
				xPred[i] += s
			}
		}
		// p_pred = A p_filt A^T + Q
		// First: tmpDD = A p_filt
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += get(aFlat, dDim, dDim, i, k) * pFilt[k*dDim+j]
				}
				tmpDD[i*dDim+j] = s
			}
		}
		// p_pred = tmpDD @ A^T + Q
		for i := 0; i < dDim; i++ {
			for j := 0; j < dDim; j++ {
				s := 0.0
				for k := 0; k < dDim; k++ {
					s += tmpDD[i*dDim+k] * get(aFlat, dDim, dDim, j, k)
				}
				pPred[i*dDim+j] = s + get(qFlat, dDim, dDim, i, j)
			}
		}
	}

	logLikOut[0] = logLik
}

// choleskyInPlace fills `l` (n×n flat) with the lower-triangular
// Cholesky factor of the symmetric PSD matrix `m` (n×n flat).
func choleskyInPlace(m, l []float64, n int) {
	for i := 0; i < n*n; i++ {
		l[i] = 0.0
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := m[i*n+j]
			for k := 0; k < j; k++ {
				s -= l[i*n+k] * l[j*n+k]
			}
			if i == j {
				l[i*n+j] = math.Sqrt(s)
			} else {
				l[i*n+j] = s / l[j*n+j]
			}
		}
	}
}

// choleskySolve solves L L^T x = b given the Cholesky factor `l`
// (n×n flat). Result in `x` (n flat).
func choleskySolve(l, b, x []float64, n int) {
	// Forward: L y = b
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for j := 0; j < i; j++ {
			s -= l[i*n+j] * y[j]
		}
		y[i] = s / l[i*n+i]
	}
	// Backward: L^T x = y
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for j := i + 1; j < n; j++ {
			s -= l[j*n+i] * x[j]
		}
		x[i] = s / l[i*n+i]
	}
}

// ─────────────────────── C ABI ───────────────────────

//export kalman_filter_c
func kalman_filter_c(
	obsPtr *C.double, ctlPtr *C.double,
	aPtr *C.double, bPtr *C.double, cPtr *C.double, dPtr *C.double,
	qPtr *C.double, rPtr *C.double,
	mu0Ptr *C.double, sigma0Ptr *C.double,
	tLen, pDim, mDim, dDim C.int,
	meansOutPtr *C.double, covsOutPtr *C.double,
	predMeansOutPtr *C.double, predCovsOutPtr *C.double,
	logLikOutPtr *C.double,
) {
	t := int(tLen)
	p := int(pDim)
	m := int(mDim)
	d := int(dDim)

	// Wrap C pointers as Go slices
	wrap := func(ptr *C.double, n int) []float64 {
		if n == 0 {
			return nil
		}
		return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
	}

	obsFlat := wrap(obsPtr, t*p)
	ctlFlat := wrap(ctlPtr, t*m)
	aFlat := wrap(aPtr, d*d)
	bFlat := wrap(bPtr, d*m)
	cFlat := wrap(cPtr, p*d)
	dFlatMat := wrap(dPtr, p*m)
	qFlat := wrap(qPtr, d*d)
	rFlat := wrap(rPtr, p*p)
	mu0 := wrap(mu0Ptr, d)
	sigma0Flat := wrap(sigma0Ptr, d*d)

	meansOut := wrap(meansOutPtr, t*d)
	covsOut := wrap(covsOutPtr, t*d*d)
	predMeansOut := wrap(predMeansOutPtr, t*d)
	predCovsOut := wrap(predCovsOutPtr, t*d*d)
	logLikOut := wrap(logLikOutPtr, 1)

	kalmanFilter(
		obsFlat, ctlFlat,
		aFlat, bFlat, cFlat, dFlatMat, qFlat, rFlat,
		mu0, sigma0Flat,
		t, p, m, d,
		meansOut, covsOut, predMeansOut, predCovsOut,
		logLikOut,
	)
}

func main() {} // required for c-shared

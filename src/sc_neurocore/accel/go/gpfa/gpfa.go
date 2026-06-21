// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go GPFA EM backend (parity with analysis/spike_stats/gpfa.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libgpfa.so`) that the Python dispatcher
// loads via ctypes. gpfa_em_c runs the GPFA EM loop from a caller-supplied
// deterministic initialisation. The linear algebra is Cholesky-based and
// structured: the marginal log-likelihood uses the Woodbury identity and the
// matrix-determinant lemma so it never forms the dense (n_obs x n_obs)
// covariance, matching the NumPy, Rust, Julia and Mojo backends within float64
// round-off.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// gpKernel builds the squared-exponential GP kernel for n time points.
func gpKernel(n int, tau float64) []float64 {
	k := make([]float64, n*n)
	tauSq := tau*tau + 1e-12
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := float64(i - j)
			k[i*n+j] = math.Exp(-0.5 * diff * diff / tauSq)
		}
	}
	return k
}

// cholesky returns the lower-triangular Cholesky factor L (row-major n*n) of the
// symmetric positive-definite matrix a (row-major), and false on a non-positive
// pivot. GPFA only factors SPD matrices, so Cholesky is the stable, ~2x cheaper
// choice over a general LU elimination.
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

// cholSolve solves M X = B for cols right-hand sides (B and X row-major n*cols)
// given the lower Cholesky factor L of M, via forward and back substitution.
func cholSolve(l []float64, n int, b []float64, cols int) []float64 {
	x := make([]float64, n*cols)
	y := make([]float64, n)
	for c := 0; c < cols; c++ {
		for i := 0; i < n; i++ {
			s := b[i*cols+c]
			for k := 0; k < i; k++ {
				s -= l[i*n+k] * y[k]
			}
			y[i] = s / l[i*n+i]
		}
		for i := n - 1; i >= 0; i-- {
			s := y[i]
			for k := i + 1; k < n; k++ {
				s -= l[k*n+i] * x[k*cols+c]
			}
			x[i*cols+c] = s / l[i*n+i]
		}
	}
	return x
}

// cholLogdet returns log|M| = 2 sum log L_ii from the Cholesky factor.
func cholLogdet(l []float64, n int) float64 {
	s := 0.0
	for i := 0; i < n; i++ {
		s += math.Log(l[i*n+i])
	}
	return 2.0 * s
}

// identity returns the n*n identity matrix, row-major.
func identity(n int) []float64 {
	e := make([]float64, n*n)
	for i := 0; i < n; i++ {
		e[i*n+i] = 1.0
	}
	return e
}

// spdInverse returns the inverse of an SPD matrix (row-major) via Cholesky.
func spdInverse(a []float64, n int) []float64 {
	l, ok := cholesky(a, n)
	if !ok {
		panic("spdInverse: matrix is not positive definite")
	}
	return cholSolve(l, n, identity(n), n)
}

// gpfaPrecision assembles M = blkdiag(K_j^-1) + A^T R^-1 A (row-major
// n_state x n_state, n_state = nLatents*nBins) and the GP prior log-determinant
// log|K|. A^T R^-1 A has the Kronecker form delta_{s,t} (C^T R^-1 C)[j,k], adding
// the constant (C^T R^-1 C)[j,k] along the time-diagonal of each (j,k) block. Each
// kernel carries a 1e-6 jitter so the regularised kernel is the model kernel
// everywhere, and is Cholesky-factored once for both inverse block and logdet.
func gpfaPrecision(c, rDiag []float64, kAll [][]float64, nNeurons, nBins, nLatents int) ([]float64, float64) {
	nState := nLatents * nBins
	rInv := make([]float64, nNeurons)
	for k := 0; k < nNeurons; k++ {
		rInv[k] = 1.0 / rDiag[k]
	}
	ctrInvC := make([]float64, nLatents*nLatents)
	for i := 0; i < nLatents; i++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nNeurons; k++ {
				s += c[k*nLatents+i] * rInv[k] * c[k*nLatents+j]
			}
			ctrInvC[i*nLatents+j] = s
		}
	}
	m := make([]float64, nState*nState)
	logdetK := 0.0
	for j := 0; j < nLatents; j++ {
		kReg := make([]float64, nBins*nBins)
		copy(kReg, kAll[j])
		for i := 0; i < nBins; i++ {
			kReg[i*nBins+i] += 1e-6
		}
		l, ok := cholesky(kReg, nBins)
		if !ok {
			panic("gpfaPrecision: GP prior is not positive definite")
		}
		logdetK += cholLogdet(l, nBins)
		kInv := cholSolve(l, nBins, identity(nBins), nBins)
		slj := j * nBins
		for i := 0; i < nBins; i++ {
			for jj := 0; jj < nBins; jj++ {
				m[(slj+i)*nState+(slj+jj)] = kInv[i*nBins+jj]
			}
		}
	}
	for j := 0; j < nLatents; j++ {
		for k := 0; k < nLatents; k++ {
			v := ctrInvC[j*nLatents+k]
			for t := 0; t < nBins; t++ {
				m[(j*nBins+t)*nState+(k*nBins+t)] += v
			}
		}
	}
	return m, logdetK
}

// eStep computes the posterior mean (flat, nLatents*nBins) and E[xx^T]. The
// posterior precision M is Cholesky-factored once; the same factor yields the mean
// (M^-1 A^T R^-1 y) and the covariance (M^-1).
func eStep(y, c, d, rDiag []float64, kAll [][]float64, nNeurons, nBins, nLatents int) ([]float64, []float64) {
	nState := nLatents * nBins
	rInv := make([]float64, nNeurons)
	for k := 0; k < nNeurons; k++ {
		rInv[k] = 1.0 / rDiag[k]
	}
	m, _ := gpfaPrecision(c, rDiag, kAll, nNeurons, nBins, nLatents)

	rhs := make([]float64, nState)
	for t := 0; t < nBins; t++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nNeurons; k++ {
				s += c[k*nLatents+j] * rInv[k] * (y[k*nBins+t] - d[k])
			}
			rhs[j*nBins+t] = s
		}
	}
	l, ok := cholesky(m, nState)
	if !ok {
		panic("eStep: precision is not positive definite")
	}
	xVec := cholSolve(l, nState, rhs, 1)
	sigma := cholSolve(l, nState, identity(nState), nState)

	xxPost := make([]float64, nLatents*nLatents)
	for t := 0; t < nBins; t++ {
		for j := 0; j < nLatents; j++ {
			xj := xVec[j*nBins+t]
			for k := 0; k < nLatents; k++ {
				xk := xVec[k*nBins+t]
				xxPost[j*nLatents+k] += xj*xk + sigma[(j*nBins+t)*nState+(k*nBins+t)]
			}
		}
	}
	return xVec, xxPost
}

// mStep updates C, d and the noise diagonal R.
func mStep(y, xPost, xxPost []float64, nNeurons, nBins, nLatents int) ([]float64, []float64, []float64) {
	dNew := make([]float64, nNeurons)
	for i := 0; i < nNeurons; i++ {
		s := 0.0
		for t := 0; t < nBins; t++ {
			s += y[i*nBins+t]
		}
		dNew[i] = s / float64(nBins)
	}
	yx := make([]float64, nNeurons*nLatents)
	for i := 0; i < nNeurons; i++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for t := 0; t < nBins; t++ {
				s += (y[i*nBins+t] - dNew[i]) * xPost[j*nBins+t]
			}
			yx[i*nLatents+j] = s
		}
	}
	xxReg := make([]float64, nLatents*nLatents)
	copy(xxReg, xxPost)
	for i := 0; i < nLatents; i++ {
		xxReg[i*nLatents+i] += 1e-8
	}
	xxInv := spdInverse(xxReg, nLatents)
	cNew := make([]float64, nNeurons*nLatents)
	for i := 0; i < nNeurons; i++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nLatents; k++ {
				s += yx[i*nLatents+k] * xxInv[k*nLatents+j]
			}
			cNew[i*nLatents+j] = s
		}
	}
	rNew := make([]float64, nNeurons)
	for i := 0; i < nNeurons; i++ {
		yyt := 0.0
		for t := 0; t < nBins; t++ {
			v := y[i*nBins+t] - dNew[i]
			yyt += v * v
		}
		yyt /= float64(nBins)
		cxy := 0.0
		for j := 0; j < nLatents; j++ {
			for t := 0; t < nBins; t++ {
				cxy += cNew[i*nLatents+j] * xPost[j*nBins+t] * (y[i*nBins+t] - dNew[i])
			}
		}
		cxy /= float64(nBins)
		rNew[i] = math.Max(yyt-cxy, 1e-6)
	}
	return cNew, dNew, rNew
}

// logLikelihood returns the exact marginal Gaussian log-likelihood via the
// Woodbury identity and the matrix-determinant lemma, routed through the
// n_state x n_state posterior precision M (never forming the dense covariance):
//
//	y^T Sigma^-1 y = y^T R^-1 y - (A^T R^-1 y)^T M^-1 (A^T R^-1 y)
//	log|Sigma|     = log|M| + log|K| + log|R_big|
func logLikelihood(y, c, d, rDiag []float64, kAll [][]float64, nNeurons, nBins, nLatents int) float64 {
	nObs := nNeurons * nBins
	nState := nLatents * nBins
	rInv := make([]float64, nNeurons)
	for k := 0; k < nNeurons; k++ {
		rInv[k] = 1.0 / rDiag[k]
	}
	m, logdetK := gpfaPrecision(c, rDiag, kAll, nNeurons, nBins, nLatents)

	rhs := make([]float64, nState)
	for t := 0; t < nBins; t++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nNeurons; k++ {
				s += c[k*nLatents+j] * rInv[k] * (y[k*nBins+t] - d[k])
			}
			rhs[j*nBins+t] = s
		}
	}
	l, ok := cholesky(m, nState)
	if !ok {
		panic("logLikelihood: precision is not positive definite")
	}
	logdetM := cholLogdet(l, nState)
	xMean := cholSolve(l, nState, rhs, 1)
	rhsXMean := 0.0
	for i := 0; i < nState; i++ {
		rhsXMean += rhs[i] * xMean[i]
	}
	yRinvY := 0.0
	for k := 0; k < nNeurons; k++ {
		for t := 0; t < nBins; t++ {
			v := y[k*nBins+t] - d[k]
			yRinvY += rInv[k] * v * v
		}
	}
	quad := yRinvY - rhsXMean
	logdetRBig := 0.0
	for k := 0; k < nNeurons; k++ {
		logdetRBig += math.Log(rDiag[k])
	}
	logdetRBig *= float64(nBins)
	logdetSigma := logdetM + logdetK + logdetRBig
	return -0.5 * (quad + logdetSigma + float64(nObs)*math.Log(2.0*math.Pi))
}

// gpfaEMFromInit runs the EM loop and returns trajectories, C, d, R_diag, log_liks.
func gpfaEMFromInit(y, c0, d0, r0 []float64, tau []float64, nNeurons, nBins, nLatents, maxIter int, tol float64) ([]float64, []float64, []float64, []float64, []float64) {
	kAll := make([][]float64, nLatents)
	for j := 0; j < nLatents; j++ {
		kAll[j] = gpKernel(nBins, tau[j])
	}
	c := make([]float64, len(c0))
	copy(c, c0)
	d := make([]float64, len(d0))
	copy(d, d0)
	r := make([]float64, len(r0))
	copy(r, r0)
	logLiks := make([]float64, 0, maxIter)
	xPost := make([]float64, nLatents*nBins)
	for iter := 0; iter < maxIter; iter++ {
		xp, xxPost := eStep(y, c, d, r, kAll, nNeurons, nBins, nLatents)
		xPost = xp
		c, d, r = mStep(y, xPost, xxPost, nNeurons, nBins, nLatents)
		ll := logLikelihood(y, c, d, r, kAll, nNeurons, nBins, nLatents)
		logLiks = append(logLiks, ll)
		if len(logLiks) > 1 && math.Abs(ll-logLiks[len(logLiks)-2]) < tol {
			break
		}
	}
	return xPost, c, d, r, logLiks
}

//export gpfa_em_c
func gpfa_em_c(
	yPtr, c0Ptr, d0Ptr, r0Ptr, tauPtr *C.double,
	nNeurons, nBins, nLatents, maxIter C.int,
	tol C.double,
	xOutPtr, paramsOutPtr, loglikOutPtr *C.double,
) {
	nn := int(nNeurons)
	nb := int(nBins)
	nl := int(nLatents)
	mi := int(maxIter)
	wrap := func(ptr *C.double, n int) []float64 {
		if n == 0 {
			return nil
		}
		return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
	}
	y := wrap(yPtr, nn*nb)
	c0 := wrap(c0Ptr, nn*nl)
	d0 := wrap(d0Ptr, nn)
	r0 := wrap(r0Ptr, nn)
	tau := wrap(tauPtr, nl)
	xOut := wrap(xOutPtr, nl*nb)
	paramsOut := wrap(paramsOutPtr, nn*nl+2*nn)
	loglikOut := wrap(loglikOutPtr, mi+1)

	xPost, c, d, r, logLiks := gpfaEMFromInit(y, c0, d0, r0, tau, nn, nb, nl, mi, float64(tol))
	copy(xOut, xPost)
	copy(paramsOut[:nn*nl], c)
	copy(paramsOut[nn*nl:nn*nl+nn], d)
	copy(paramsOut[nn*nl+nn:nn*nl+2*nn], r)
	loglikOut[0] = float64(len(logLiks))
	copy(loglikOut[1:1+len(logLiks)], logLiks)
}

func main() {} // required for c-shared

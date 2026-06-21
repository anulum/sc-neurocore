// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go GPFA EM backend (parity with analysis/spike_stats/gpfa.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libgpfa.so`) that the Python dispatcher
// loads via ctypes. The function gpfa_em_c runs the GPFA EM loop from a
// caller-supplied deterministic initialisation and produces trajectories,
// parameters and exact marginal Gaussian log-likelihoods identical to the
// NumPy, Rust and Julia backends within float64 round-off.
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

// matSolve solves A x = b (A is n*n, b is n*m) via Gauss-Jordan elimination.
func matSolve(a, b []float64, n, m int) []float64 {
	w := n + m
	aug := make([]float64, n*w)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i*w+j] = a[i*n+j]
		}
		for j := 0; j < m; j++ {
			aug[i*w+n+j] = b[i*m+j]
		}
	}
	for col := 0; col < n; col++ {
		maxRow := col
		maxVal := math.Abs(aug[col*w+col])
		for row := col + 1; row < n; row++ {
			if v := math.Abs(aug[row*w+col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}
		if maxVal < 1e-30 {
			continue
		}
		if maxRow != col {
			for k := 0; k < w; k++ {
				aug[col*w+k], aug[maxRow*w+k] = aug[maxRow*w+k], aug[col*w+k]
			}
		}
		pivot := aug[col*w+col]
		for k := 0; k < w; k++ {
			aug[col*w+k] /= pivot
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row*w+col]
			for k := 0; k < w; k++ {
				aug[row*w+k] -= factor * aug[col*w+k]
			}
		}
	}
	x := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			x[i*m+j] = aug[i*w+n+j]
		}
	}
	return x
}

// identity returns the n*n identity matrix, row-major.
func identity(n int) []float64 {
	e := make([]float64, n*n)
	for i := 0; i < n; i++ {
		e[i*n+i] = 1.0
	}
	return e
}

// matSlogdet returns the sign and natural log absolute determinant via LU.
func matSlogdet(a []float64, n int) (float64, float64) {
	m := make([]float64, len(a))
	copy(m, a)
	sign := 1.0
	logAbs := 0.0
	for col := 0; col < n; col++ {
		maxRow := col
		maxVal := math.Abs(m[col*n+col])
		for row := col + 1; row < n; row++ {
			if v := math.Abs(m[row*n+col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}
		if maxVal < 1e-300 {
			return 0.0, math.Inf(-1)
		}
		if maxRow != col {
			for k := 0; k < n; k++ {
				m[col*n+k], m[maxRow*n+k] = m[maxRow*n+k], m[col*n+k]
			}
			sign = -sign
		}
		pivot := m[col*n+col]
		if pivot < 0 {
			sign = -sign
		}
		logAbs += math.Log(math.Abs(pivot))
		for row := col + 1; row < n; row++ {
			factor := m[row*n+col] / pivot
			for k := col; k < n; k++ {
				m[row*n+k] -= factor * m[col*n+k]
			}
		}
	}
	return sign, logAbs
}

// eStep computes the posterior mean (flat, n_latents*n_bins) and E[xx^T].
func eStep(y, c, d, rDiag []float64, kAll [][]float64, nNeurons, nBins, nLatents int) ([]float64, []float64) {
	kt := nLatents * nBins
	rInv := make([]float64, nNeurons)
	for k := 0; k < nNeurons; k++ {
		rInv[k] = 1.0 / (rDiag[k] + 1e-10)
	}

	ctRinvC := make([]float64, nLatents*nLatents)
	for i := 0; i < nLatents; i++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nNeurons; k++ {
				s += c[k*nLatents+i] * rInv[k] * c[k*nLatents+j]
			}
			ctRinvC[i*nLatents+j] = s
		}
	}
	ctRinv := make([]float64, nLatents*nNeurons)
	for i := 0; i < nLatents; i++ {
		for k := 0; k < nNeurons; k++ {
			ctRinv[i*nNeurons+k] = c[k*nLatents+i] * rInv[k]
		}
	}

	prec := make([]float64, kt*kt)
	eyeBins := identity(nBins)
	for j := 0; j < nLatents; j++ {
		slj := j * nBins
		kReg := make([]float64, nBins*nBins)
		copy(kReg, kAll[j])
		for i := 0; i < nBins; i++ {
			kReg[i*nBins+i] += 1e-6
		}
		kInv := matSolve(kReg, eyeBins, nBins, nBins)
		for i := 0; i < nBins; i++ {
			for jj := 0; jj < nBins; jj++ {
				diag := 0.0
				if i == jj {
					diag = 1.0
				}
				prec[(slj+i)*kt+(slj+jj)] = kInv[i*nBins+jj] + ctRinvC[j*nLatents+j]*diag
			}
		}
		for k := 0; k < nLatents; k++ {
			if k != j {
				slk := k * nBins
				for i := 0; i < nBins; i++ {
					prec[(slj+i)*kt+(slk+i)] = ctRinvC[j*nLatents+k]
				}
			}
		}
	}

	rhs := make([]float64, kt)
	for t := 0; t < nBins; t++ {
		for j := 0; j < nLatents; j++ {
			s := 0.0
			for k := 0; k < nNeurons; k++ {
				s += ctRinv[j*nNeurons+k] * (y[k*nBins+t] - d[k])
			}
			rhs[j*nBins+t] = s
		}
	}
	for i := 0; i < kt; i++ {
		prec[i*kt+i] += 1e-8
	}

	xVec := matSolve(prec, rhs, kt, 1)
	sigmaPost := matSolve(prec, identity(kt), kt, kt)

	xxPost := make([]float64, nLatents*nLatents)
	for t := 0; t < nBins; t++ {
		for j := 0; j < nLatents; j++ {
			xj := xVec[j*nBins+t]
			for k := 0; k < nLatents; k++ {
				xk := xVec[k*nBins+t]
				xxPost[j*nLatents+k] += xj*xk + sigmaPost[(j*nBins+t)*kt+(k*nBins+t)]
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
	xxInv := matSolve(xxReg, identity(nLatents), nLatents, nLatents)
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

// logLikelihood returns the exact marginal Gaussian log-likelihood.
func logLikelihood(y, c, d, rDiag []float64, kAll [][]float64, nNeurons, nBins, nLatents int) float64 {
	nObs := nNeurons * nBins
	nState := nLatents * nBins
	a := make([]float64, nObs*nState)
	for t := 0; t < nBins; t++ {
		rowStart := t * nNeurons
		for j := 0; j < nLatents; j++ {
			col := j*nBins + t
			for i := 0; i < nNeurons; i++ {
				a[(rowStart+i)*nState+col] = c[i*nLatents+j]
			}
		}
	}
	kBig := make([]float64, nState*nState)
	for j := 0; j < nLatents; j++ {
		for ii := 0; ii < nBins; ii++ {
			for jj := 0; jj < nBins; jj++ {
				kBig[(j*nBins+ii)*nState+(j*nBins+jj)] = kAll[j][ii*nBins+jj]
			}
		}
	}
	ak := make([]float64, nObs*nState)
	for i := 0; i < nObs; i++ {
		for j := 0; j < nState; j++ {
			s := 0.0
			for k := 0; k < nState; k++ {
				s += a[i*nState+k] * kBig[k*nState+j]
			}
			ak[i*nState+j] = s
		}
	}
	cov := make([]float64, nObs*nObs)
	for i := 0; i < nObs; i++ {
		for j := 0; j < nObs; j++ {
			s := 0.0
			for k := 0; k < nState; k++ {
				s += ak[i*nState+k] * a[j*nState+k]
			}
			cov[i*nObs+j] = s
		}
	}
	for t := 0; t < nBins; t++ {
		for i := 0; i < nNeurons; i++ {
			idx := t*nNeurons + i
			cov[idx*nObs+idx] += rDiag[i]
		}
	}
	for i := 0; i < nObs; i++ {
		cov[i*nObs+i] += 1e-8
	}
	yc := make([]float64, nObs)
	for t := 0; t < nBins; t++ {
		for i := 0; i < nNeurons; i++ {
			yc[t*nNeurons+i] = y[i*nBins+t] - d[i]
		}
	}
	sol := matSolve(cov, yc, nObs, 1)
	quad := 0.0
	for i := 0; i < nObs; i++ {
		quad += yc[i] * sol[i]
	}
	_, logdet := matSlogdet(cov, nObs)
	return -0.5 * (quad + logdet + float64(nObs)*math.Log(2.0*math.Pi))
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

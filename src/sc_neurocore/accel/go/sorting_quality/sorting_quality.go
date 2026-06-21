// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go sorting-quality backend (parity with
// analysis/spike_stats/sorting_quality.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libsorting_quality.so`) that the Python
// dispatcher loads via ctypes. isolation_distance_c and l_ratio_c evaluate the
// Mahalanobis cluster-quality metrics (Harris et al. 2001; Schmitzer-Torbert et
// al. 2005). The squared Mahalanobis distance is computed through the Cholesky
// factor of the regularised cluster covariance — the covariance is never
// inverted explicitly — matching the NumPy, Rust, Julia and Mojo backends within
// float64 round-off.
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

// cholesky returns the lower Cholesky factor L (row-major) of the SPD matrix a
// (n x n). The cluster covariance is jitter-regularised, so the diagonal pivot
// stays positive; the clamp is a defensive floor for fully degenerate inputs.
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

// featureCovariance returns the d x d unbiased (ddof=1) covariance of data
// (n x d, row-major) over its rows, with eps jitter on the diagonal.
func featureCovariance(data []float64, n, d int, eps float64) []float64 {
	mu := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			mu[j] += data[i*d+j]
		}
	}
	for j := 0; j < d; j++ {
		mu[j] /= float64(n)
	}
	denom := math.Max(float64(n-1), 1.0)
	cov := make([]float64, d*d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			dj := data[i*d+j] - mu[j]
			for k := j; k < d; k++ {
				cov[j*d+k] += dj * (data[i*d+k] - mu[k])
			}
		}
	}
	for j := 0; j < d; j++ {
		for k := j; k < d; k++ {
			cov[j*d+k] /= denom
			cov[k*d+j] = cov[j*d+k]
		}
		cov[j*d+j] += eps
	}
	return cov
}

// mahalanobisSq returns the squared Mahalanobis distances of each row of points
// (nPts x d) from the cluster mean, using the Cholesky factor of the cluster
// covariance and a forward substitution (mah = ||L^-1 (x-mu)||^2).
func mahalanobisSq(cluster []float64, nCluster int, points []float64, nPts, d int) []float64 {
	mu := make([]float64, d)
	for i := 0; i < nCluster; i++ {
		for j := 0; j < d; j++ {
			mu[j] += cluster[i*d+j]
		}
	}
	for j := 0; j < d; j++ {
		mu[j] /= float64(nCluster)
	}
	l := cholesky(featureCovariance(cluster, nCluster, d, 1e-8), d)

	out := make([]float64, nPts)
	z := make([]float64, d)
	for p := 0; p < nPts; p++ {
		for j := 0; j < d; j++ {
			s := points[p*d+j] - mu[j]
			for k := 0; k < j; k++ {
				s -= l[j*d+k] * z[k]
			}
			z[j] = s / l[j*d+j]
		}
		m := 0.0
		for j := 0; j < d; j++ {
			m += z[j] * z[j]
		}
		out[p] = m
	}
	return out
}

// isolationDistance is the Mahalanobis isolation distance (Harris et al. 2001).
func isolationDistance(cluster []float64, nCluster int, noise []float64, nNoise, d int) float64 {
	if nCluster < 2 || nNoise < nCluster {
		return math.NaN()
	}
	mah := mahalanobisSq(cluster, nCluster, noise, nNoise, d)
	sort.Float64s(mah)
	if nCluster-1 < len(mah) {
		return mah[nCluster-1]
	}
	return mah[len(mah)-1]
}

// lRatio is the L-ratio cluster-quality metric (Schmitzer-Torbert et al. 2005).
func lRatio(cluster []float64, nCluster int, noise []float64, nNoise, d int) float64 {
	if nCluster < 2 || nNoise == 0 {
		return math.NaN()
	}
	mah := mahalanobisSq(cluster, nCluster, noise, nNoise, d)
	df := float64(d)
	sum := 0.0
	for _, m := range mah {
		v := math.Exp(-0.5 * (math.Max(m, 1e-10) - df))
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		sum += v
	}
	return sum / float64(nCluster)
}

//export isolation_distance_c
func isolation_distance_c(
	clusterPtr *C.double, nCluster C.int,
	noisePtr *C.double, nNoise, nFeatures C.int,
) C.double {
	nc := int(nCluster)
	nn := int(nNoise)
	d := int(nFeatures)
	if d == 0 {
		return C.double(math.NaN())
	}
	cluster := unsafe.Slice((*float64)(unsafe.Pointer(clusterPtr)), nc*d)
	noise := unsafe.Slice((*float64)(unsafe.Pointer(noisePtr)), nn*d)
	return C.double(isolationDistance(cluster, nc, noise, nn, d))
}

//export l_ratio_c
func l_ratio_c(
	clusterPtr *C.double, nCluster C.int,
	noisePtr *C.double, nNoise, nFeatures C.int,
) C.double {
	nc := int(nCluster)
	nn := int(nNoise)
	d := int(nFeatures)
	if d == 0 {
		return C.double(math.NaN())
	}
	cluster := unsafe.Slice((*float64)(unsafe.Pointer(clusterPtr)), nc*d)
	noise := unsafe.Slice((*float64)(unsafe.Pointer(noisePtr)), nn*d)
	return C.double(lRatio(cluster, nc, noise, nn, d))
}

func main() {} // required for c-shared

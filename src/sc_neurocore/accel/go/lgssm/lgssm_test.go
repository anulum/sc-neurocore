// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go LGSSM Kalman filter tests

package main

import (
	"math"
	"testing"
)

func TestKalmanFilterScalarPredictionCovariance(t *testing.T) {
	obs := []float64{1.0, 0.5}
	controls := []float64{}
	a := []float64{0.9}
	b := []float64{}
	c := []float64{1.0}
	d := []float64{}
	q := []float64{0.1}
	r := []float64{0.5}
	mu0 := []float64{0.0}
	sigma0 := []float64{1.0}

	means := make([]float64, 2)
	covs := make([]float64, 2)
	predMeans := make([]float64, 2)
	predCovs := make([]float64, 2)
	logLik := make([]float64, 1)

	kalmanFilter(
		obs,
		controls,
		a,
		b,
		c,
		d,
		q,
		r,
		mu0,
		sigma0,
		2,
		1,
		0,
		1,
		means,
		covs,
		predMeans,
		predCovs,
		logLik,
	)

	if math.Abs(covs[0]-1.0/3.0) > 1e-12 {
		t.Fatalf("filtered covariance[0] = %.16f, want %.16f", covs[0], 1.0/3.0)
	}
	expectedNextPredCov := 0.9*0.9*(1.0/3.0) + 0.1
	if math.Abs(predCovs[1]-expectedNextPredCov) > 1e-12 {
		t.Fatalf(
			"predicted covariance[1] = %.16f, want %.16f",
			predCovs[1],
			expectedNextPredCov,
		)
	}
}

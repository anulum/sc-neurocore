// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for traub_miles

package services

import (
	"errors"
	"math"
)

// TraubMilesNeuronState holds the neuron state
type TraubMilesNeuronState struct {
	V          float64
	M          float64
	H          float64
	N          float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	Dt         float64
	VThreshold float64
}

// NewTraubMilesNeuron creates a new TraubMilesNeuron neuron with default parameters
func NewTraubMilesNeuron() *TraubMilesNeuronState {
	return &TraubMilesNeuronState{
		V:          -67.0,
		M:          0.05,
		H:          0.6,
		N:          0.3,
		GNa:        100.0,
		GK:         80.0,
		GL:         0.1,
		ENa:        50.0,
		EK:         -100.0,
		EL:         -67.0,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

func finiteTraubMiles(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteTraubMilesGate(value float64) bool {
	return finiteTraubMiles(value) && value >= 0.0 && value <= 1.0
}

// ValidateTraubMiles checks runtime state and physical parameters.
func ValidateTraubMiles(s *TraubMilesNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteTraubMiles(s.V) &&
		finiteTraubMilesGate(s.M) &&
		finiteTraubMilesGate(s.H) &&
		finiteTraubMilesGate(s.N) &&
		finiteTraubMiles(s.GNa) && s.GNa >= 0.0 &&
		finiteTraubMiles(s.GK) && s.GK >= 0.0 &&
		finiteTraubMiles(s.GL) && s.GL >= 0.0 &&
		finiteTraubMiles(s.ENa) &&
		finiteTraubMiles(s.EK) &&
		finiteTraubMiles(s.EL) &&
		finiteTraubMiles(s.Dt) && s.Dt > 0.0 &&
		finiteTraubMiles(s.VThreshold)
}

func traubMilesRates(v float64) (float64, float64, float64, float64, float64, float64, error) {
	d := v + 54.0
	am := 8.0
	if math.Abs(d) > 1.0e-6 {
		am = 0.32 * d / (1.0 - math.Exp(-d/4.0))
	}
	d2 := v + 27.0
	bm := 5.6
	if math.Abs(d2) > 1.0e-6 {
		bm = 0.28 * d2 / (math.Exp(d2/5.0) - 1.0)
	}
	ah := 0.128 * math.Exp(-(v+50.0)/18.0)
	bh := 4.0 / (1.0 + math.Exp(-(v+27.0)/5.0))
	d3 := v + 52.0
	an := 0.32
	if math.Abs(d3) > 1.0e-6 {
		an = 0.032 * d3 / (1.0 - math.Exp(-d3/5.0))
	}
	bn := 0.5 * math.Exp(-(v+57.0)/40.0)
	for _, rate := range []float64{am, bm, ah, bh, an, bn} {
		if !finiteTraubMiles(rate) || rate < 0.0 {
			return 0, 0, 0, 0, 0, 0, errors.New("invalid Traub-Miles rate")
		}
	}
	return am, bm, ah, bh, an, bn, nil
}

func traubMilesDerivatives(s *TraubMilesNeuronState, v float64, m float64, h float64, n float64, iExt float64) (float64, float64, float64, float64, error) {
	if !finiteTraubMiles(v) || !finiteTraubMilesGate(m) || !finiteTraubMilesGate(h) || !finiteTraubMilesGate(n) {
		return 0, 0, 0, 0, errors.New("invalid Traub-Miles derivative state")
	}
	am, bm, ah, bh, an, bn, err := traubMilesRates(v)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	dm := am*(1.0-m) - bm*m
	dh := ah*(1.0-h) - bh*h
	dn := an*(1.0-n) - bn*n
	iNa := s.GNa * math.Pow(m, 3.0) * h * (v - s.ENa)
	iK := s.GK * math.Pow(n, 4.0) * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := -iNa - iK - iL + iExt
	for _, value := range []float64{dv, dm, dh, dn, iNa, iK, iL} {
		if !finiteTraubMiles(value) {
			return 0, 0, 0, 0, errors.New("invalid Traub-Miles derivative")
		}
	}
	return dv, dm, dh, dn, nil
}

func traubMilesRK4Substep(s *TraubMilesNeuronState, v float64, m float64, h float64, n float64, iExt float64) (float64, float64, float64, float64, error) {
	k1V, k1M, k1H, k1N, err := traubMilesDerivatives(s, v, m, h, n, iExt)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	k2V, k2M, k2H, k2N, err := traubMilesDerivatives(s, v+0.5*s.Dt*k1V, m+0.5*s.Dt*k1M, h+0.5*s.Dt*k1H, n+0.5*s.Dt*k1N, iExt)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	k3V, k3M, k3H, k3N, err := traubMilesDerivatives(s, v+0.5*s.Dt*k2V, m+0.5*s.Dt*k2M, h+0.5*s.Dt*k2H, n+0.5*s.Dt*k2N, iExt)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	k4V, k4M, k4H, k4N, err := traubMilesDerivatives(s, v+s.Dt*k3V, m+s.Dt*k3M, h+s.Dt*k3H, n+s.Dt*k3N, iExt)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	nextV := v + s.Dt*(k1V+2.0*k2V+2.0*k3V+k4V)/6.0
	nextM := m + s.Dt*(k1M+2.0*k2M+2.0*k3M+k4M)/6.0
	nextH := h + s.Dt*(k1H+2.0*k2H+2.0*k3H+k4H)/6.0
	nextN := n + s.Dt*(k1N+2.0*k2N+2.0*k3N+k4N)/6.0
	if !finiteTraubMiles(nextV) || !finiteTraubMilesGate(nextM) || !finiteTraubMilesGate(nextH) || !finiteTraubMilesGate(nextN) {
		return 0, 0, 0, 0, errors.New("invalid Traub-Miles candidate state")
	}
	return nextV, nextM, nextH, nextN, nil
}

// Step advances the neuron by one timestep
func (s *TraubMilesNeuronState) Step(iExt float64) (int, error) {
	if !ValidateTraubMiles(s) {
		return 0, errors.New("invalid Traub-Miles runtime state")
	}
	if !finiteTraubMiles(iExt) {
		return 0, errors.New("invalid Traub-Miles external current")
	}

	vPrev := s.V
	v := s.V
	m := s.M
	h := s.H
	n := s.N
	for substep := 0; substep < 10; substep++ {
		nextV, nextM, nextH, nextN, err := traubMilesRK4Substep(s, v, m, h, n, iExt)
		if err != nil {
			return 0, err
		}
		v = nextV
		m = nextM
		h = nextH
		n = nextN
	}

	s.V = v
	s.M = m
	s.H = h
	s.N = n
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateTraubMilesNeuron runs the neuron for n steps
func SimulateTraubMilesNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTraubMilesNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

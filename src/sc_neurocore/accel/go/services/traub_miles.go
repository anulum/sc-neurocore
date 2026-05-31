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
		am, bm, ah, bh, an, bn, err := traubMilesRates(v)
		if err != nil {
			return 0, err
		}
		nextM := m + (am*(1.0-m)-bm*m)*s.Dt
		nextH := h + (ah*(1.0-h)-bh*h)*s.Dt
		nextN := n + (an*(1.0-n)-bn*n)*s.Dt
		if !finiteTraubMilesGate(nextM) || !finiteTraubMilesGate(nextH) || !finiteTraubMilesGate(nextN) {
			return 0, errors.New("invalid Traub-Miles gate candidate")
		}
		iNa := s.GNa * nextM * nextM * nextM * nextH * (v - s.ENa)
		iK := s.GK * math.Pow(nextN, 4.0) * (v - s.EK)
		iL := s.GL * (v - s.EL)
		nextV := v + (-iNa-iK-iL+iExt)*s.Dt
		if !finiteTraubMiles(nextV) {
			return 0, errors.New("invalid Traub-Miles voltage candidate")
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

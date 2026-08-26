// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for TTypeCaNeuron

package services

import (
	"errors"
	"math"
)

// TTypeCaNeuronState holds the complete WB+IT state and configuration.
type TTypeCaNeuronState struct {
	V          float64
	H          float64
	N          float64
	S          float64
	GNa        float64
	GK         float64
	GT         float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewTTypeCaNeuron creates a TTypeCaNeuron with the canonical repository defaults.
func NewTTypeCaNeuron() *TTypeCaNeuronState {
	return &TTypeCaNeuronState{
		V: -65.0, H: 0.6, N: 0.32, S: 0.9,
		GNa: 35.0, GK: 9.0, GT: 0.1, GL: 0.2,
		ENa: 55.0, EK: -90.0, ECa: 120.0, EL: -65.0,
		CM: 1.0, Phi: 5.0,
		Dt: 0.5, VThreshold: -20.0, Gain: 1.0, SubSteps: 50,
	}
}

func ttypeFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func ttypeBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidTTypeCaNeuron enforces the public descriptor and runtime safety bounds.
func ValidTTypeCaNeuron(s *TTypeCaNeuronState) bool {
	if s == nil || !ttypeFinite(
		s.V, s.H, s.N, s.S, s.GNa, s.GK, s.GT, s.GL, s.ENa, s.EK,
		s.ECa, s.EL, s.CM, s.Phi, s.Dt, s.VThreshold, s.Gain,
	) {
		return false
	}
	return ttypeBetween(s.V, -100.0, 60.0) &&
		ttypeBetween(s.H, 0.0, 1.0) && ttypeBetween(s.N, 0.0, 1.0) &&
		ttypeBetween(s.S, 0.0, 1.0) &&
		ttypeBetween(s.GNa, 0.0, 200.0) && ttypeBetween(s.GK, 0.0, 100.0) &&
		ttypeBetween(s.GT, 0.0, 20.0) && ttypeBetween(s.GL, 0.0, 5.0) &&
		ttypeBetween(s.ENa, 30.0, 70.0) && ttypeBetween(s.EK, -100.0, -70.0) &&
		ttypeBetween(s.ECa, 60.0, 150.0) && ttypeBetween(s.EL, -80.0, -40.0) &&
		ttypeBetween(s.CM, 0.5, 2.0) && ttypeBetween(s.Phi, 0.5, 10.0) &&
		s.Dt > 0.0 && s.Dt <= 1.0 && ttypeBetween(s.VThreshold, -20.0, 20.0) &&
		ttypeBetween(s.Gain, 0.0, 10.0) && s.SubSteps >= 1 && s.SubSteps <= 10000
}

func ttypeSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *TTypeCaNeuronState) TryStep(iExt float64) (int, error) {
	if !ttypeFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !ValidTTypeCaNeuron(s) {
		return 0, errors.New("T-type state and parameters must satisfy the public bounds")
	}

	next := *s
	input := next.Gain * iExt
	subDt := next.Dt / float64(next.SubSteps)
	fired := 0
	for index := 0; index < next.SubSteps; index++ {
		v := next.V
		alphaM := ttypeSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := ttypeSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		mTInf := 1.0 / (1.0 + math.Exp(-(v+52.0)/5.0))
		sInf := 1.0 / (1.0 + math.Exp((v+81.0)/4.0))
		tauS := 30.0 + 100.0/(1.0+math.Exp((v+75.0)/10.0))

		next.H += subDt * next.Phi * (alphaH*(1.0-next.H) - betaH*next.H)
		next.N += subDt * next.Phi * (alphaN*(1.0-next.N) - betaN*next.N)
		next.S += subDt * (sInf - next.S) / tauS
		iNa := next.GNa * math.Pow(mInf, 3.0) * next.H * (v - next.ENa)
		iK := next.GK * math.Pow(next.N, 4.0) * (v - next.EK)
		iT := next.GT * mTInf * mTInf * next.S * (v - next.ECa)
		iL := next.GL * (v - next.EL)
		next.V += subDt * (-iNa - iK - iT - iL + input) / next.CM
		if !ttypeFinite(next.V, next.H, next.N, next.S) {
			return 0, errors.New("T-type candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			fired = 1
			next.V = -65.0
			next.S *= 0.3
		}
	}

	next.V = math.Max(-100.0, math.Min(60.0, next.V))
	next.H = math.Max(0.0, math.Min(1.0, next.H))
	next.N = math.Max(0.0, math.Min(1.0, next.N))
	next.S = math.Max(0.0, math.Min(1.0, next.S))
	*s = next
	return fired, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *TTypeCaNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *TTypeCaNeuronState) Reset() {
	s.V, s.H, s.N, s.S = -65.0, 0.6, 0.32, 0.9
}

// SimulateTTypeCaNeuron runs the neuron for n steps.
func SimulateTTypeCaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTTypeCaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.V
	}
	return trace, spikes
}

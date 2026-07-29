// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for IhNeuron

package services

import (
	"errors"
	"math"
)

// IhNeuronState holds the complete WB+HCN state and configuration.
type IhNeuronState struct {
	V          float64
	H          float64
	N          float64
	R          float64
	GNa        float64
	GK         float64
	GH         float64
	GL         float64
	ENa        float64
	EK         float64
	EH         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewIhNeuron creates an IhNeuron with the canonical repository defaults.
func NewIhNeuron() *IhNeuronState {
	return &IhNeuronState{
		V: -65.0, H: 0.6, N: 0.32, R: 0.1,
		GNa: 35.0, GK: 9.0, GH: 0.15, GL: 0.2,
		ENa: 55.0, EK: -90.0, EH: -40.0, EL: -65.0,
		CM: 1.0, Phi: 5.0, Dt: 0.5, VThreshold: -20.0,
		Gain: 1.0, SubSteps: 50,
	}
}

func ihFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func ihBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidIhNeuron enforces the public descriptor and runtime safety bounds.
func ValidIhNeuron(s *IhNeuronState) bool {
	if s == nil || !ihFinite(
		s.V, s.H, s.N, s.R, s.GNa, s.GK, s.GH, s.GL, s.ENa, s.EK,
		s.EH, s.EL, s.CM, s.Phi, s.Dt, s.VThreshold, s.Gain,
	) {
		return false
	}
	return ihBetween(s.V, -100.0, 60.0) &&
		ihBetween(s.H, 0.0, 1.0) && ihBetween(s.N, 0.0, 1.0) && ihBetween(s.R, 0.0, 1.0) &&
		ihBetween(s.GNa, 0.0, 200.0) && ihBetween(s.GK, 0.0, 100.0) &&
		ihBetween(s.GH, 0.0, 5.0) && ihBetween(s.GL, 0.0, 5.0) &&
		ihBetween(s.ENa, 30.0, 70.0) && ihBetween(s.EK, -100.0, -70.0) &&
		ihBetween(s.EH, -50.0, 0.0) && ihBetween(s.EL, -80.0, -40.0) &&
		ihBetween(s.CM, 0.5, 2.0) && ihBetween(s.Phi, 0.5, 10.0) &&
		s.Dt > 0.0 && s.Dt <= 1.0 && ihBetween(s.VThreshold, -20.0, 20.0) &&
		ihBetween(s.Gain, 0.0, 10.0) && s.SubSteps >= 1 && s.SubSteps <= 10000
}

func ihSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *IhNeuronState) TryStep(iExt float64) (int, error) {
	if !ihFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !ValidIhNeuron(s) {
		return 0, errors.New("Ih state and parameters must satisfy the public bounds")
	}

	next := *s
	input := next.Gain * iExt
	subDt := next.Dt / float64(next.SubSteps)
	fired := 0
	for index := 0; index < next.SubSteps; index++ {
		v := next.V
		alphaM := ihSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := ihSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		rInf := 1.0 / (1.0 + math.Exp((v+80.0)/10.0))
		tauR := 100.0 + 200.0/(1.0+math.Exp((v+70.0)/10.0))

		next.H += subDt * next.Phi * (alphaH*(1.0-next.H) - betaH*next.H)
		next.N += subDt * next.Phi * (alphaN*(1.0-next.N) - betaN*next.N)
		next.R += subDt * (rInf - next.R) / tauR
		iNa := next.GNa * math.Pow(mInf, 3.0) * next.H * (v - next.ENa)
		iK := next.GK * math.Pow(next.N, 4.0) * (v - next.EK)
		iH := next.GH * next.R * (v - next.EH)
		iL := next.GL * (v - next.EL)
		next.V += subDt * (-iNa - iK - iH - iL + input) / next.CM
		if !ihFinite(next.V, next.H, next.N, next.R) {
			return 0, errors.New("Ih candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			fired = 1
			next.V = -65.0
		}
	}

	next.V = math.Max(-100.0, math.Min(60.0, next.V))
	next.H = math.Max(0.0, math.Min(1.0, next.H))
	next.N = math.Max(0.0, math.Min(1.0, next.N))
	next.R = math.Max(0.0, math.Min(1.0, next.R))
	*s = next
	return fired, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *IhNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *IhNeuronState) Reset() {
	s.V, s.H, s.N, s.R = -65.0, 0.6, 0.32, 0.1
}

// SimulateIhNeuron runs the neuron for n steps.
func SimulateIhNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewIhNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.V
	}
	return trace, spikes
}

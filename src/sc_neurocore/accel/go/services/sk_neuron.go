// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for SKNeuron

package services

import (
	"errors"
	"math"
)

// SKNeuronState holds the complete WB+SK state and configuration.
type SKNeuronState struct {
	V          float64
	H          float64
	N          float64
	Ca         float64
	GNa        float64
	GK         float64
	GSk        float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	TauCa      float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewSKNeuron creates an SKNeuron with the canonical repository defaults.
func NewSKNeuron() *SKNeuronState {
	return &SKNeuronState{
		V: -65.0, H: 0.6, N: 0.32, Ca: 0.0,
		GNa: 35.0, GK: 9.0, GSk: 2.0, GL: 0.1,
		ENa: 55.0, EK: -90.0, EL: -65.0,
		CM: 1.0, Phi: 5.0, TauCa: 150.0,
		Dt: 0.5, VThreshold: -20.0, Gain: 1.0, SubSteps: 50,
	}
}

func skFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func skBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidSKNeuron enforces the public descriptor and runtime safety bounds.
func ValidSKNeuron(s *SKNeuronState) bool {
	if s == nil || !skFinite(
		s.V, s.H, s.N, s.Ca, s.GNa, s.GK, s.GSk, s.GL, s.ENa, s.EK,
		s.EL, s.CM, s.Phi, s.TauCa, s.Dt, s.VThreshold, s.Gain,
	) {
		return false
	}
	return skBetween(s.V, -100.0, 60.0) &&
		skBetween(s.H, 0.0, 1.0) && skBetween(s.N, 0.0, 1.0) &&
		s.Ca >= 0.0 &&
		skBetween(s.GNa, 0.0, 200.0) && skBetween(s.GK, 0.0, 100.0) &&
		skBetween(s.GSk, 0.0, 50.0) && skBetween(s.GL, 0.0, 5.0) &&
		skBetween(s.ENa, 30.0, 70.0) && skBetween(s.EK, -100.0, -70.0) &&
		skBetween(s.EL, -80.0, -40.0) &&
		skBetween(s.CM, 0.5, 2.0) && skBetween(s.Phi, 0.5, 10.0) &&
		skBetween(s.TauCa, 10.0, 2000.0) &&
		s.Dt > 0.0 && s.Dt <= 1.0 && skBetween(s.VThreshold, -20.0, 20.0) &&
		skBetween(s.Gain, 0.0, 10.0) && s.SubSteps >= 1 && s.SubSteps <= 10000
}

func skSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *SKNeuronState) TryStep(iExt float64) (int, error) {
	if !skFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !ValidSKNeuron(s) {
		return 0, errors.New("SK state and parameters must satisfy the public bounds")
	}

	next := *s
	input := next.Gain * iExt
	subDt := next.Dt / float64(next.SubSteps)
	fired := 0
	for index := 0; index < next.SubSteps; index++ {
		v := next.V
		alphaM := skSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := skSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		ca2 := next.Ca * next.Ca
		skInf := ca2 / (ca2 + 0.25)

		next.Ca += subDt * (-next.Ca / next.TauCa)

		next.H += subDt * next.Phi * (alphaH*(1.0-next.H) - betaH*next.H)
		next.N += subDt * next.Phi * (alphaN*(1.0-next.N) - betaN*next.N)
		iNa := next.GNa * math.Pow(mInf, 3.0) * next.H * (v - next.ENa)
		iK := next.GK * math.Pow(next.N, 4.0) * (v - next.EK)
		iSk := next.GSk * skInf * (v - next.EK)
		iL := next.GL * (v - next.EL)
		next.V += subDt * (-iNa - iK - iSk - iL + input) / next.CM
		if !skFinite(next.V, next.H, next.N, next.Ca) {
			return 0, errors.New("SK candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			fired = 1
			next.V = -65.0
			next.Ca += 0.2
		}
	}

	next.V = math.Max(-100.0, math.Min(60.0, next.V))
	next.H = math.Max(0.0, math.Min(1.0, next.H))
	next.N = math.Max(0.0, math.Min(1.0, next.N))
	next.Ca = math.Max(0.0, next.Ca)
	*s = next
	return fired, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *SKNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *SKNeuronState) Reset() {
	s.V, s.H, s.N, s.Ca = -65.0, 0.6, 0.32, 0.0
}

// SimulateSKNeuron runs the neuron for n steps.
func SimulateSKNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSKNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.V
	}
	return trace, spikes
}

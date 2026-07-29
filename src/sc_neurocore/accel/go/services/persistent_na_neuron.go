// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for PersistentNaNeuron

package services

import (
	"errors"
	"math"
)

// PersistentNaNeuronState holds the complete WB+INaP state and configuration.
type PersistentNaNeuronState struct {
	V          float64
	H          float64
	N          float64
	P          float64
	GNa        float64
	GNap       float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewPersistentNaNeuron creates a neuron with canonical repository defaults.
func NewPersistentNaNeuron() *PersistentNaNeuronState {
	return &PersistentNaNeuronState{
		V: -65.0, H: 0.6, N: 0.32, P: 0.0,
		GNa: 35.0, GNap: 0.15, GK: 9.0, GL: 0.3,
		ENa: 55.0, EK: -90.0, EL: -65.0,
		CM: 1.0, Phi: 5.0, Dt: 0.5, VThreshold: -20.0,
		Gain: 1.0, SubSteps: 50,
	}
}

func persistentNaFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func persistentNaBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidPersistentNaNeuron enforces the public descriptor and safety bounds.
func ValidPersistentNaNeuron(s *PersistentNaNeuronState) bool {
	if s == nil || !persistentNaFinite(
		s.V, s.H, s.N, s.P, s.GNa, s.GNap, s.GK, s.GL, s.ENa, s.EK,
		s.EL, s.CM, s.Phi, s.Dt, s.VThreshold, s.Gain,
	) {
		return false
	}
	return persistentNaBetween(s.V, -100.0, 60.0) &&
		persistentNaBetween(s.H, 0.0, 1.0) && persistentNaBetween(s.N, 0.0, 1.0) &&
		persistentNaBetween(s.P, 0.0, 1.0) &&
		persistentNaBetween(s.GNa, 0.0, 200.0) && persistentNaBetween(s.GNap, 0.0, 20.0) &&
		persistentNaBetween(s.GK, 0.0, 100.0) && persistentNaBetween(s.GL, 0.0, 5.0) &&
		persistentNaBetween(s.ENa, 30.0, 70.0) && persistentNaBetween(s.EK, -100.0, -70.0) &&
		persistentNaBetween(s.EL, -80.0, -40.0) && persistentNaBetween(s.CM, 0.5, 2.0) &&
		persistentNaBetween(s.Phi, 0.5, 10.0) && s.Dt > 0.0 && s.Dt <= 1.0 &&
		persistentNaBetween(s.VThreshold, -20.0, 20.0) && persistentNaBetween(s.Gain, 0.0, 10.0) &&
		s.SubSteps >= 1 && s.SubSteps <= 10000
}

func persistentNaSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *PersistentNaNeuronState) TryStep(iExt float64) (int, error) {
	if !persistentNaFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !ValidPersistentNaNeuron(s) {
		return 0, errors.New("PersistentNa state and parameters must satisfy the public bounds")
	}

	next := *s
	input := next.Gain * iExt
	subDt := next.Dt / float64(next.SubSteps)
	fired := 0
	for index := 0; index < next.SubSteps; index++ {
		v := next.V
		alphaM := persistentNaSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := persistentNaSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		pInf := 1.0 / (1.0 + math.Exp(-(v+48.0)/5.0))
		tauP := 10.0 + 40.0/(1.0+math.Pow((v+48.0)/10.0, 2.0))

		next.H += subDt * next.Phi * (alphaH*(1.0-next.H) - betaH*next.H)
		next.N += subDt * next.Phi * (alphaN*(1.0-next.N) - betaN*next.N)
		next.P += subDt * (pInf - next.P) / tauP
		iNa := next.GNa * math.Pow(mInf, 3.0) * next.H * (v - next.ENa)
		iNap := next.GNap * next.P * (v - next.ENa)
		iK := next.GK * math.Pow(next.N, 4.0) * (v - next.EK)
		iL := next.GL * (v - next.EL)
		next.V += subDt * (-iNa - iNap - iK - iL + input) / next.CM
		if !persistentNaFinite(next.V, next.H, next.N, next.P) {
			return 0, errors.New("PersistentNa candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			fired = 1
			next.V = -65.0
		}
	}

	next.V = math.Max(-100.0, math.Min(60.0, next.V))
	next.H = math.Max(0.0, math.Min(1.0, next.H))
	next.N = math.Max(0.0, math.Min(1.0, next.N))
	next.P = math.Max(0.0, math.Min(1.0, next.P))
	*s = next
	return fired, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *PersistentNaNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *PersistentNaNeuronState) Reset() {
	s.V, s.H, s.N, s.P = -65.0, 0.6, 0.32, 0.0
}

// SimulatePersistentNaNeuron runs the neuron for n steps.
func SimulatePersistentNaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPersistentNaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.V
	}
	return trace, spikes
}

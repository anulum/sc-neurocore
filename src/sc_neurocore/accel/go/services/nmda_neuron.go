// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for NMDANeuron

package services

import (
	"errors"
	"math"
)

// NMDANeuronState holds the complete WB+NMDA state and configuration.
type NMDANeuronState struct {
	V          float64
	H          float64
	N          float64
	SNmda      float64
	GNa        float64
	GK         float64
	GNmda      float64
	GL         float64
	ENa        float64
	EK         float64
	ENmda      float64
	EL         float64
	CM         float64
	Phi        float64
	MgConc     float64
	TauRise    float64
	TauDecay   float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewNMDANeuron creates an NMDANeuron with the canonical repository defaults.
func NewNMDANeuron() *NMDANeuronState {
	return &NMDANeuronState{
		V: -65.0, H: 0.6, N: 0.32, SNmda: 0.0,
		GNa: 35.0, GK: 9.0, GNmda: 0.5, GL: 0.1,
		ENa: 55.0, EK: -90.0, ENmda: 0.0, EL: -65.0,
		CM: 1.0, Phi: 5.0, MgConc: 1.0,
		TauRise: 10.0, TauDecay: 100.0,
		Dt: 0.5, VThreshold: -20.0, Gain: 1.0, SubSteps: 50,
	}
}

func nmdaFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func nmdaBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidNMDANeuron enforces the public descriptor and runtime safety bounds.
func ValidNMDANeuron(s *NMDANeuronState) bool {
	if s == nil || !nmdaFinite(
		s.V, s.H, s.N, s.SNmda, s.GNa, s.GK, s.GNmda, s.GL, s.ENa, s.EK,
		s.ENmda, s.EL, s.CM, s.Phi, s.MgConc, s.TauRise, s.TauDecay,
		s.Dt, s.VThreshold, s.Gain,
	) {
		return false
	}
	return nmdaBetween(s.V, -100.0, 60.0) &&
		nmdaBetween(s.H, 0.0, 1.0) && nmdaBetween(s.N, 0.0, 1.0) &&
		nmdaBetween(s.SNmda, 0.0, 1.0) &&
		nmdaBetween(s.GNa, 0.0, 200.0) && nmdaBetween(s.GK, 0.0, 100.0) &&
		nmdaBetween(s.GNmda, 0.0, 20.0) && nmdaBetween(s.GL, 0.0, 5.0) &&
		nmdaBetween(s.ENa, 30.0, 70.0) && nmdaBetween(s.EK, -100.0, -70.0) &&
		nmdaBetween(s.ENmda, -10.0, 10.0) && nmdaBetween(s.EL, -80.0, -40.0) &&
		nmdaBetween(s.CM, 0.5, 2.0) && nmdaBetween(s.Phi, 0.5, 10.0) &&
		nmdaBetween(s.MgConc, 0.0, 5.0) &&
		nmdaBetween(s.TauRise, 0.1, 20.0) && nmdaBetween(s.TauDecay, 10.0, 500.0) &&
		s.Dt > 0.0 && s.Dt <= 1.0 && nmdaBetween(s.VThreshold, -20.0, 20.0) &&
		nmdaBetween(s.Gain, 0.0, 10.0) && s.SubSteps >= 1 && s.SubSteps <= 10000
}

func nmdaSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *NMDANeuronState) TryStep(iExt float64) (int, error) {
	if !nmdaFinite(iExt) {
		return 0, errors.New("current must be finite")
	}
	if !ValidNMDANeuron(s) {
		return 0, errors.New("NMDA state and parameters must satisfy the public bounds")
	}

	next := *s
	input := next.Gain * iExt
	subDt := next.Dt / float64(next.SubSteps)
	fired := 0

	drive := 0.0
	if input > 0.0 {
		drive = input / (input + 5.0)
	}
	tau := next.TauDecay
	if drive > next.SNmda {
		tau = next.TauRise
	}
	ds := (drive - next.SNmda) / tau
	next.SNmda += next.Dt * ds
	next.SNmda = math.Max(0.0, math.Min(1.0, next.SNmda))

	for index := 0; index < next.SubSteps; index++ {
		v := next.V
		alphaM := nmdaSafeRate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * math.Exp(-(v+58.0)/20.0)
		betaH := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		alphaN := nmdaSafeRate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * math.Exp(-(v+44.0)/80.0)
		mgBlock := 1.0 / (1.0 + (next.MgConc/3.57)*math.Exp(-0.062*v))

		next.H += subDt * next.Phi * (alphaH*(1.0-next.H) - betaH*next.H)
		next.N += subDt * next.Phi * (alphaN*(1.0-next.N) - betaN*next.N)
		iNa := next.GNa * math.Pow(mInf, 3.0) * next.H * (v - next.ENa)
		iK := next.GK * math.Pow(next.N, 4.0) * (v - next.EK)
		iNmda := next.GNmda * next.SNmda * mgBlock * (v - next.ENmda)
		iL := next.GL * (v - next.EL)
		next.V += subDt * (-iNa - iK - iNmda - iL + input) / next.CM
		if !nmdaFinite(next.V, next.H, next.N) {
			return 0, errors.New("NMDA candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			fired = 1
			next.V = -65.0
		}
	}

	next.V = math.Max(-100.0, math.Min(60.0, next.V))
	next.H = math.Max(0.0, math.Min(1.0, next.H))
	next.N = math.Max(0.0, math.Min(1.0, next.N))
	*s = next
	return fired, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *NMDANeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *NMDANeuronState) Reset() {
	s.V, s.H, s.N, s.SNmda = -65.0, 0.6, 0.32, 0.0
}

// SimulateNMDANeuron runs the neuron for n steps.
func SimulateNMDANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewNMDANeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.V
	}
	return trace, spikes
}

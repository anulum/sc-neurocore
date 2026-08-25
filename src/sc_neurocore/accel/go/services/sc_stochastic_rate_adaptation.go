// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for SC stochastic rate adaptation

package services

import "math"

// SCStochasticRateAdaptationNeuronState holds the project neuron state.
type SCStochasticRateAdaptationNeuronState struct {
	A      float64
	FMax   float64
	Beta   float64
	IHalf  float64
	TauA   float64
	DeltaA float64
	Dt     float64
	Rng    float64
}

// NewSCStochasticRateAdaptationNeuron creates the count-neutral project model.
func NewSCStochasticRateAdaptationNeuron() *SCStochasticRateAdaptationNeuronState {
	return &SCStochasticRateAdaptationNeuronState{A: 0.0, FMax: 200.0, Beta: 0.1, IHalf: 5.0, TauA: 100.0, DeltaA: 0.5, Dt: 1.0, Rng: 0.0}
}

// Valid reports whether the stochastic rate-adaptation state is admissible.
func (s SCStochasticRateAdaptationNeuronState) Valid() bool {
	return finite(s.A) && s.A >= 0.0 &&
		finite(s.FMax) && s.FMax > 0.0 &&
		finite(s.Beta) && s.Beta > 0.0 &&
		finite(s.IHalf) &&
		finite(s.TauA) && s.TauA > 0.0 &&
		finite(s.DeltaA) && s.DeltaA >= 0.0 &&
		finite(s.Dt) && s.Dt > 0.0 &&
		finite(s.Rng) && s.Rng >= 0.0 && s.Rng < 1.0
}

// FOnset evaluates the overflow-stable Benda-Herz onset f-I curve.
func (s SCStochasticRateAdaptationNeuronState) FOnset(x float64) float64 {
	z := s.Beta * (x - s.IHalf)
	if math.IsInf(z, 1) {
		return s.FMax
	}
	if math.IsInf(z, -1) {
		return 0.0
	}
	if z >= 0.0 {
		return s.FMax / (1.0 + math.Exp(-z))
	}
	expZ := math.Exp(z)
	return s.FMax * expZ / (1.0 + expZ)
}

func (s SCStochasticRateAdaptationNeuronState) adaptationRHS(a, iExt float64) (float64, float64, bool) {
	if !finite(a) || a < 0.0 {
		return 0.0, 0.0, false
	}
	rate := s.FOnset(iExt - a)
	if !finite(rate) || rate < 0.0 || rate > s.FMax {
		return 0.0, 0.0, false
	}
	return -a/s.TauA + s.DeltaA*rate, rate, true
}

// RK4Candidate returns the next adaptation value and Bernoulli spike probability.
func (s SCStochasticRateAdaptationNeuronState) RK4Candidate(iExt float64) (float64, float64, bool) {
	k1, r1, ok := s.adaptationRHS(s.A, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k2, r2, ok := s.adaptationRHS(s.A+0.5*s.Dt*k1, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k3, r3, ok := s.adaptationRHS(s.A+0.5*s.Dt*k2, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k4, r4, ok := s.adaptationRHS(s.A+s.Dt*k3, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	nextA := s.A + (s.Dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
	averageRate := (r1 + 2.0*r2 + 2.0*r3 + r4) / 6.0
	hazard := averageRate * s.Dt / 1000.0
	if !finite(nextA) || nextA < 0.0 || !finite(hazard) || hazard < 0.0 {
		return 0.0, 0.0, false
	}
	p := -math.Expm1(-hazard)
	if !finite(p) || p < 0.0 || p > 1.0 {
		return 0.0, 0.0, false
	}
	return nextA, p, true
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *SCStochasticRateAdaptationNeuronState) Step(iExt float64) int {
	if !finite(iExt) || !s.Valid() {
		return 0
	}

	nextA, p, ok := s.RK4Candidate(iExt)
	if !ok {
		return 0
	}

	s.A = nextA
	if s.Rng < p {
		return 1
	}
	return 0
}

// Reset clears dynamic adaptation while preserving parameters.
func (s *SCStochasticRateAdaptationNeuronState) Reset() {
	s.A = 0.0
}

// SimulateSCStochasticRateAdaptationNeuron runs the project neuron for n steps.
func SimulateSCStochasticRateAdaptationNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSCStochasticRateAdaptationNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.A
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

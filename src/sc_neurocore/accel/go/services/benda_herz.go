// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for benda_herz

package services

import "math"

// BendaHerzNeuronState holds the neuron state.
type BendaHerzNeuronState struct {
	A      float64
	FMax   float64
	Beta   float64
	IHalf  float64
	TauA   float64
	DeltaA float64
	Dt     float64
	Rng    float64
}

// NewBendaHerzNeuron creates a new BendaHerzNeuron neuron with default parameters.
func NewBendaHerzNeuron() *BendaHerzNeuronState {
	return &BendaHerzNeuronState{A: 0.0, FMax: 200.0, Beta: 0.1, IHalf: 5.0, TauA: 100.0, DeltaA: 0.5, Dt: 1.0, Rng: 0.0}
}

// Valid reports whether the stochastic rate-adaptation state is admissible.
func (s BendaHerzNeuronState) Valid() bool {
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
func (s BendaHerzNeuronState) FOnset(x float64) float64 {
	z := s.Beta * (x - s.IHalf)
	if z >= 0.0 {
		return s.FMax / (1.0 + math.Exp(-z))
	}
	expZ := math.Exp(z)
	return s.FMax * expZ / (1.0 + expZ)
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *BendaHerzNeuronState) Step(iExt float64) int {
	if !finite(iExt) || !s.Valid() {
		return 0
	}

	rate := s.FOnset(iExt - s.A)
	p := rate * s.Dt / 1000.0
	if !finite(rate) || !finite(p) || p > 1.0 {
		return 0
	}
	nextA := s.A + (-s.A/s.TauA+s.DeltaA*rate)*s.Dt
	if !finite(nextA) || nextA < 0.0 {
		return 0
	}

	s.A = nextA
	if s.Rng < p {
		return 1
	}
	return 0
}

// Reset clears dynamic adaptation while preserving parameters.
func (s *BendaHerzNeuronState) Reset() {
	s.A = 0.0
}

// SimulateBendaHerzNeuron runs the neuron for n steps.
func SimulateBendaHerzNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBendaHerzNeuron()
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

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for quadratic_if

package services

import (
	"errors"
	"math"
)

// QuadraticIFNeuronState holds the neuron state.
type QuadraticIFNeuronState struct {
	V      float64
	VReset float64
	VPeak  float64
	Dt     float64
}

// NewQuadraticIFNeuron creates a new QuadraticIFNeuron neuron with default parameters.
func NewQuadraticIFNeuron() *QuadraticIFNeuronState {
	return &QuadraticIFNeuronState{
		V:      -1.0,
		VReset: -1.0,
		VPeak:  1.0,
		Dt:     0.01,
	}
}

func quadraticIFFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Valid reports whether the state satisfies the QIF integration contract.
func (s QuadraticIFNeuronState) Valid() bool {
	return quadraticIFFinite(s.V, s.VReset, s.VPeak, s.Dt) &&
		s.V < s.VPeak &&
		s.VReset < s.VPeak &&
		s.Dt > 0.0
}

func (s QuadraticIFNeuronState) exactCandidate(iExt float64) (float64, bool) {
	if iExt > 0.0 {
		rootI := math.Sqrt(iExt)
		phase := math.Atan(s.V / rootI)
		peakPhase := math.Atan(s.VPeak / rootI)
		nextPhase := phase + rootI*s.Dt
		if nextPhase >= peakPhase || nextPhase >= math.Pi/2.0 {
			return s.VReset, true
		}
		return rootI * math.Tan(nextPhase), false
	}
	if iExt == 0.0 {
		denominator := 1.0 - s.V*s.Dt
		if denominator <= 0.0 {
			return s.VReset, true
		}
		nextV := s.V / denominator
		if nextV >= s.VPeak {
			return s.VReset, true
		}
		return nextV, false
	}

	rootI := math.Sqrt(-iExt)
	if math.Abs(s.V+rootI) <= 1.0e-15 {
		return s.V, false
	}
	numeratorRatio := (s.V - rootI) / (s.V + rootI)
	evolvedRatio := numeratorRatio * math.Exp(2.0*rootI*s.Dt)
	denominator := 1.0 - evolvedRatio
	if (numeratorRatio < 1.0 && evolvedRatio >= 1.0) || math.Abs(denominator) <= 1.0e-15 {
		return s.VReset, true
	}
	nextV := rootI * (1.0 + evolvedRatio) / denominator
	if nextV >= s.VPeak {
		return s.VReset, true
	}
	return nextV, false
}

// Step advances the neuron by one exact constant-current QIF flow step. Invalid inputs do not mutate state.
func (s *QuadraticIFNeuronState) Step(iExt float64) (int, error) {
	if !quadraticIFFinite(iExt) || !s.Valid() {
		return 0, ErrQuadraticIFInvalidState
	}

	nextV, spiked := s.exactCandidate(iExt)
	if !quadraticIFFinite(nextV) {
		return 0, ErrQuadraticIFNonFiniteUpdate
	}

	s.V = nextV
	if spiked {
		return 1, nil
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *QuadraticIFNeuronState) Reset() {
	s.V = s.VReset
}

// SimulateQuadraticIFNeuron runs the neuron for n steps.
func SimulateQuadraticIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewQuadraticIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrQuadraticIFInvalidState    = errors.New("quadratic-if state/current must be finite and well-formed")
	ErrQuadraticIFNonFiniteUpdate = errors.New("quadratic-if exact-flow update became non-finite")
)

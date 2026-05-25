// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lapicque

package services

import "errors"

// LapicqueNeuronState holds the neuron state.
type LapicqueNeuronState struct {
	V          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	Tau        float64
	Resistance float64
	Dt         float64
}

// NewLapicqueNeuron creates a new LapicqueNeuron neuron with default parameters.
func NewLapicqueNeuron() *LapicqueNeuronState {
	return &LapicqueNeuronState{
		V:          0.0,
		VRest:      0.0,
		VReset:     0.0,
		VThreshold: 1.0,
		Tau:        20.0,
		Resistance: 1.0,
		Dt:         1.0,
	}
}

// Valid reports whether the state satisfies the Lapicque RC integration contract.
func (s LapicqueNeuronState) Valid() bool {
	return finite(s.V) &&
		finite(s.VRest) &&
		finite(s.VReset) &&
		finite(s.VThreshold) && s.VThreshold > s.VRest && s.VThreshold > s.VReset &&
		s.V < s.VThreshold &&
		finite(s.Tau) && s.Tau > 0.0 &&
		finite(s.Resistance) && s.Resistance > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *LapicqueNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) {
		return 0, errors.New("lapicque input current must be finite")
	}
	if !s.Valid() {
		return 0, errors.New("lapicque state must satisfy finite positive-RC threshold contract")
	}

	dv := (-(s.V - s.VRest) + s.Resistance*iExt) / s.Tau * s.Dt
	nextV := s.V + dv
	if !finite(dv) || !finite(nextV) {
		return 0, errors.New("lapicque voltage increment must remain finite")
	}

	s.V = nextV
	if s.V >= s.VThreshold {
		s.V = s.VReset
		return 1, nil
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *LapicqueNeuronState) Reset() {
	s.V = s.VRest
}

// SimulateLapicqueNeuron runs the neuron for n steps.
func SimulateLapicqueNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLapicqueNeuron()
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

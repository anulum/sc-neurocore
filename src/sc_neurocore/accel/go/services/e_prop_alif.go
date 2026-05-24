// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for e_prop_alif

package services

import (
	"math"
)

// EPropALIFNeuronState holds the neuron state
type EPropALIFNeuronState struct {
	V              float64
	A              float64
	ETrace         float64
	TauM           float64
	TauA           float64
	VThresholdBase float64
	Beta           float64
	VReset         float64
	Dt             float64
	AlphaM         float64
	AlphaA         float64
}

// NewEPropALIFNeuron creates a new EPropALIFNeuron neuron with default parameters
func NewEPropALIFNeuron() *EPropALIFNeuronState {
	return &EPropALIFNeuronState{
		V:              0.0,
		A:              0.0,
		ETrace:         0.0,
		TauM:           20.0,
		TauA:           200.0,
		VThresholdBase: 1.0,
		Beta:           0.07,
		VReset:         0.0,
		Dt:             1.0,
		AlphaM:         math.Exp(-1.0 / 20.0),
		AlphaA:         math.Exp(-1.0 / 200.0),
	}
}

// Step advances the neuron by one timestep
func (s *EPropALIFNeuronState) Step(iExt float64) int {
	if !s.Valid() || !isFiniteEPropALIF(iExt) {
		return 0
	}

	s.V = s.AlphaM*s.V + iExt
	threshold := s.VThresholdBase + s.Beta*s.A
	psi := math.Max(0.0, 1.0-math.Abs(s.V-threshold)) * 0.3
	s.ETrace = s.AlphaA*s.ETrace + psi
	if s.V >= threshold {
		s.V = s.VReset
		s.A = s.AlphaA*s.A + 1.0
		return 1
	}
	s.A *= s.AlphaA
	return 0
}

// Valid returns true when the state satisfies the e-prop ALIF physics contract.
func (s *EPropALIFNeuronState) Valid() bool {
	return isFiniteEPropALIF(s.V) &&
		isFiniteEPropALIF(s.A) &&
		s.A >= 0.0 &&
		isFiniteEPropALIF(s.ETrace) &&
		isFiniteEPropALIF(s.TauM) &&
		s.TauM > 0.0 &&
		isFiniteEPropALIF(s.TauA) &&
		s.TauA > 0.0 &&
		isFiniteEPropALIF(s.VThresholdBase) &&
		isFiniteEPropALIF(s.Beta) &&
		s.Beta >= 0.0 &&
		isFiniteEPropALIF(s.VReset) &&
		isFiniteEPropALIF(s.Dt) &&
		s.Dt > 0.0 &&
		s.Dt <= s.TauM &&
		s.Dt <= s.TauA &&
		s.VThresholdBase > s.VReset &&
		isFiniteEPropALIF(s.AlphaM) &&
		s.AlphaM > 0.0 &&
		s.AlphaM < 1.0 &&
		isFiniteEPropALIF(s.AlphaA) &&
		s.AlphaA > 0.0 &&
		s.AlphaA < 1.0
}

func (s *EPropALIFNeuronState) Reset() {
	s.V = s.VReset
	s.A = 0.0
	s.ETrace = 0.0
}

func isFiniteEPropALIF(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// SimulateEPropALIFNeuron runs the neuron for n steps
func SimulateEPropALIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewEPropALIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

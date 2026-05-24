// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for energy_lif

package services

import (
	"math"
)

// EnergyLIFNeuronState holds the neuron state
type EnergyLIFNeuronState struct {
	V          float64
	Epsilon    float64
	VRest      float64
	VReset     float64
	VThreshold float64
	TauM       float64
	TauE       float64
	Alpha      float64
	Epsilon0   float64
	Resistance float64
	Dt         float64
}

// NewEnergyLIFNeuron creates a new EnergyLIFNeuron neuron with default parameters
func NewEnergyLIFNeuron() *EnergyLIFNeuronState {
	return &EnergyLIFNeuronState{
		V:          -70.0,
		Epsilon:    1.0,
		VRest:      -70.0,
		VReset:     -70.0,
		VThreshold: -50.0,
		TauM:       10.0,
		TauE:       500.0,
		Alpha:      0.1,
		Epsilon0:   1.0,
		Resistance: 1.0,
		Dt:         1.0,
	}
}

// Step advances the neuron by one timestep
func (s *EnergyLIFNeuronState) Step(iExt float64) int {
	if !s.Valid() || !isFiniteEnergyLIF(iExt) {
		return 0
	}

	effectiveR := s.Resistance * s.Epsilon
	s.V += (-(s.V - s.VRest) + effectiveR*iExt) / s.TauM * s.Dt
	s.Epsilon += (s.Epsilon0 - s.Epsilon) / s.TauE * s.Dt
	if s.V >= s.VThreshold && s.Epsilon > 0.1 {
		s.V = s.VReset
		s.Epsilon -= s.Alpha
		if s.Epsilon < 0.0 {
			s.Epsilon = 0.0
		}
		return 1
	}
	return 0
}

// Valid returns true when the state satisfies the energy-LIF physics contract.
func (s *EnergyLIFNeuronState) Valid() bool {
	return isFiniteEnergyLIF(s.V) &&
		isFiniteEnergyLIF(s.Epsilon) &&
		s.Epsilon >= 0.0 &&
		isFiniteEnergyLIF(s.VRest) &&
		isFiniteEnergyLIF(s.VReset) &&
		isFiniteEnergyLIF(s.VThreshold) &&
		isFiniteEnergyLIF(s.TauM) &&
		s.TauM > 0.0 &&
		isFiniteEnergyLIF(s.TauE) &&
		s.TauE > 0.0 &&
		isFiniteEnergyLIF(s.Alpha) &&
		s.Alpha >= 0.0 &&
		isFiniteEnergyLIF(s.Epsilon0) &&
		s.Epsilon0 >= 0.0 &&
		isFiniteEnergyLIF(s.Resistance) &&
		s.Resistance > 0.0 &&
		isFiniteEnergyLIF(s.Dt) &&
		s.Dt > 0.0 &&
		s.Epsilon <= s.Epsilon0 &&
		s.Dt <= s.TauM &&
		s.Dt <= s.TauE &&
		s.VThreshold > s.VRest &&
		s.VThreshold > s.VReset
}

func isFiniteEnergyLIF(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// SimulateEnergyLIFNeuron runs the neuron for n steps
func SimulateEnergyLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewEnergyLIFNeuron()
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

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for leaky_compete_fire

package services

import (
	"errors"
	"math"
)

// LeakyCompeteFireNeuronState holds the neuron state
type LeakyCompeteFireNeuronState struct {
	NUnits     int
	V          []float64
	Tau        float64
	VThreshold float64
	WInh       float64
	Dt         float64
}

// NewLeakyCompeteFireNeuron creates a new LeakyCompeteFireNeuron neuron with default parameters
func NewLeakyCompeteFireNeuron() *LeakyCompeteFireNeuronState {
	return &LeakyCompeteFireNeuronState{
		NUnits:     4,
		V:          []float64{0.0, 0.0, 0.0, 0.0},
		Tau:        10.0,
		VThreshold: 1.0,
		WInh:       0.5,
		Dt:         1.0,
	}
}

// Step advances the neuron by one timestep
func (s *LeakyCompeteFireNeuronState) Step(currents interface{}) ([]int, error) {
	currentValues, err := normaliseLCFCurrents(currents, s.NUnits)
	if err != nil {
		return nil, err
	}
	if err := validateLeakyCompeteFire(s); err != nil {
		return nil, err
	}
	decay := math.Exp(-s.Dt / s.Tau)
	nextV := make([]float64, s.NUnits)
	for i := 0; i < s.NUnits; i++ {
		nextV[i] = currentValues[i] + (s.V[i]-currentValues[i])*decay
		if !lcfFinite(nextV[i]) {
			return nil, errors.New("LCF exact relaxation produced a non-finite candidate")
		}
	}
	spikes := make([]int, s.NUnits)
	for i := 0; i < s.NUnits; i++ {
		if nextV[i] >= s.VThreshold {
			spikes[i] = 1
			nextV[i] = 0.0
			for j := 0; j < s.NUnits; j++ {
				if j != i {
					nextV[j] = math.Max(0.0, nextV[j]-s.WInh)
				}
			}
		}
	}
	s.V = nextV
	return spikes, nil
}

// SimulateLeakyCompeteFireNeuron runs the neuron for n steps
func SimulateLeakyCompeteFireNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLeakyCompeteFireNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			break
		}
		trace[t] = s.V[0]
		for _, spike := range result {
			spikes += spike
		}
	}
	return trace, spikes
}

func validateLeakyCompeteFire(s *LeakyCompeteFireNeuronState) error {
	if s == nil {
		return errors.New("LCF state must not be nil")
	}
	if s.NUnits <= 0 {
		return errors.New("LCF n_units must be positive")
	}
	if len(s.V) != s.NUnits {
		return errors.New("LCF voltage vector length must match n_units")
	}
	if !lcfPositive(s.Tau) || !lcfPositive(s.Dt) {
		return errors.New("LCF tau and dt must be finite and positive")
	}
	if !lcfFinite(s.VThreshold) || !lcfFinite(s.WInh) || s.WInh < 0.0 {
		return errors.New("LCF threshold and inhibition weight must be finite and valid")
	}
	for _, voltage := range s.V {
		if !lcfFinite(voltage) {
			return errors.New("LCF voltage vector must contain only finite values")
		}
	}
	return nil
}

func normaliseLCFCurrents(currents interface{}, nUnits int) ([]float64, error) {
	if nUnits <= 0 {
		return nil, errors.New("LCF n_units must be positive")
	}
	values := make([]float64, nUnits)
	switch currentSet := currents.(type) {
	case float64:
		for i := range values {
			values[i] = currentSet
		}
	case int:
		value := float64(currentSet)
		for i := range values {
			values[i] = value
		}
	case []float64:
		if len(currentSet) != nUnits {
			return nil, errors.New("LCF currents must match n_units")
		}
		copy(values, currentSet)
	default:
		return nil, errors.New("LCF currents must be a scalar or []float64")
	}
	for _, current := range values {
		if !lcfFinite(current) {
			return nil, errors.New("LCF currents must contain only finite values")
		}
	}
	return values, nil
}

func lcfPositive(value float64) bool {
	return lcfFinite(value) && value > 0.0
}

func lcfFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

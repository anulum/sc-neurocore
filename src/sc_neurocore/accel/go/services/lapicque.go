// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lapicque

package services

import (
	"errors"
	"math"
)

// LapicqueNeuronState holds the source polarization or preserved SC LIF state.
type LapicqueNeuronState struct {
	V                      float64
	VRest                  float64
	VReset                 float64
	VThreshold             float64
	Tau                    float64
	Resistance             float64
	Dt                     float64
	Capacitance            float64
	SeriesResistance       float64
	PolarizationResistance float64
	Excited                bool
	SourceProfile          bool
}

// NewLapicqueNeuron creates a new LapicqueNeuron neuron with default parameters.
func NewLapicqueNeuron() *LapicqueNeuronState {
	return &LapicqueNeuronState{
		V:                      0.0,
		VRest:                  0.0,
		VReset:                 0.0,
		VThreshold:             1.0,
		Tau:                    20.0,
		Resistance:             1.0,
		Dt:                     1.0,
		Capacitance:            1.1,
		SeriesResistance:       10.0,
		PolarizationResistance: 1.0,
	}
}

// NewLapicque1907 returns the normalized one-shot source polarization profile.
func NewLapicque1907() *LapicqueNeuronState {
	s := NewLapicqueNeuron()
	s.Dt = 0.01
	s.SourceProfile = true
	return s
}

// Valid reports whether the state satisfies the Lapicque RC integration contract.
func (s LapicqueNeuronState) Valid() bool {
	if !finiteLapicque(s.V) || !finiteLapicque(s.VThreshold) || s.VThreshold <= 0.0 ||
		!finiteLapicque(s.Dt) || s.Dt <= 0.0 {
		return false
	}
	if s.SourceProfile {
		return (s.Excited || s.V < s.VThreshold) &&
			finiteLapicque(s.Capacitance) && s.Capacitance > 0.0 &&
			finiteLapicque(s.SeriesResistance) && s.SeriesResistance > 0.0 &&
			finiteLapicque(s.PolarizationResistance) && s.PolarizationResistance > 0.0
	}
	return !s.Excited && finiteLapicque(s.VRest) && finiteLapicque(s.VReset) &&
		s.VThreshold > s.VRest && s.VThreshold > s.VReset && s.V < s.VThreshold &&
		finiteLapicque(s.Tau) && s.Tau > 0.0 &&
		finiteLapicque(s.Resistance) && s.Resistance > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *LapicqueNeuronState) Step(iExt float64) (int, error) {
	if !finiteLapicque(iExt) {
		return 0, errors.New("lapicque input current must be finite")
	}
	if !s.Valid() {
		return 0, errors.New("lapicque state must satisfy finite positive-RC threshold contract")
	}

	vInf := s.VRest + s.Resistance*iExt
	decay := math.Exp(-s.Dt / s.Tau)
	if s.SourceProfile {
		totalResistance := s.SeriesResistance + s.PolarizationResistance
		beta := s.Capacitance * s.SeriesResistance * s.PolarizationResistance / totalResistance
		vInf = iExt * s.PolarizationResistance / totalResistance
		decay = math.Exp(-s.Dt / beta)
	}
	nextV := vInf + (s.V-vInf)*decay
	if !finiteLapicque(vInf) || !finiteLapicque(decay) || !finiteLapicque(nextV) {
		return 0, errors.New("lapicque voltage candidate must remain finite")
	}

	if s.SourceProfile {
		event := !s.Excited && nextV >= s.VThreshold
		s.V = nextV
		if event {
			s.Excited = true
			return 1, nil
		}
		return 0, nil
	}

	s.V = nextV
	if nextV >= s.VThreshold {
		s.V = s.VReset
		return 1, nil
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *LapicqueNeuronState) Reset() {
	if s.SourceProfile {
		s.V = 0.0
	} else {
		s.V = s.VRest
	}
	s.Excited = false
}

func finiteLapicque(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
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

// SimulateLapicqueComplete executes a candidate batch and returns complete events.
// The receiver is not mutated when validation or any candidate step fails.
func SimulateLapicqueComplete(
	initial LapicqueNeuronState,
	nSteps int,
	drive float64,
) ([]float64, []uint8, LapicqueNeuronState, error) {
	if nSteps < 0 || !finiteLapicque(drive) || !initial.Valid() {
		return nil, nil, initial, errors.New("invalid Lapicque batch contract")
	}
	candidate := initial
	trace := make([]float64, nSteps)
	events := make([]uint8, nSteps)
	for index := 0; index < nSteps; index++ {
		event, err := candidate.Step(drive)
		if err != nil {
			return nil, nil, initial, err
		}
		trace[index] = candidate.V
		events[index] = uint8(event)
	}
	return trace, events, candidate, nil
}

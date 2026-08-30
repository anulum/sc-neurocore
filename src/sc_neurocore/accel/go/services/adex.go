// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for adex

package services

import (
	"errors"
	"math"
)

var (
	ErrAdExInvalidInput    = errors.New("adex input current must be finite")
	ErrAdExInvalidSteps    = errors.New("adex step count must be non-negative")
	ErrAdExInvalidState    = errors.New("adex state parameters must be finite with positive delta_t, tau, tau_w, c_m, and dt")
	ErrAdExNonFiniteUpdate = errors.New("adex integrator update must remain finite")
)

// AdExNeuronState holds the complete maintained baseline-Euler state and parameters.
type AdExNeuronState struct {
	V          float64
	W          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	VRh        float64
	DeltaT     float64
	Tau        float64
	TauW       float64
	A          float64
	B          float64
	CM         float64
	Dt         float64
}

// NewAdExNeuron creates an AdEx neuron with the maintained Python defaults.
func NewAdExNeuron() *AdExNeuronState {
	return &AdExNeuronState{
		V:          -65.0,
		W:          0.0,
		VRest:      -65.0,
		VReset:     -68.0,
		VThreshold: -50.0,
		VRh:        -55.0,
		DeltaT:     2.0,
		Tau:        20.0,
		TauW:       100.0,
		A:          0.5,
		B:          7.0,
		CM:         200.0,
		Dt:         0.1,
	}
}

func adExFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *AdExNeuronState) Valid() bool {
	return adExFinite(s.V, s.W, s.VRest, s.VReset, s.VThreshold, s.VRh, s.DeltaT, s.Tau, s.TauW, s.A, s.B, s.CM, s.Dt) &&
		s.DeltaT > 0.0 &&
		s.Tau > 0.0 &&
		s.TauW > 0.0 &&
		s.CM > 0.0 &&
		s.Dt > 0.0
}

// Step advances the neuron by one timestep
func (s *AdExNeuronState) Step(iExt float64) (int, error) {
	if !adExFinite(iExt) {
		return 0, ErrAdExInvalidInput
	}
	if !s.Valid() {
		return 0, ErrAdExInvalidState
	}

	arg := (s.V - s.VRh) / s.DeltaT
	if arg < -20.0 {
		arg = -20.0
	} else if arg > 20.0 {
		arg = 20.0
	}
	expTerm := s.DeltaT * math.Exp(arg)
	dv := (-(s.V-s.VRest)+expTerm)/s.Tau + (-s.W+iExt)/s.CM
	dw := (s.A*(s.V-s.VRest) - s.W) / s.TauW
	nextV := s.V + dv*s.Dt
	nextW := s.W + dw*s.Dt
	if !adExFinite(expTerm, dv, dw, nextV, nextW) {
		return 0, ErrAdExNonFiniteUpdate
	}
	if nextV >= s.VThreshold {
		spikeW := nextW + s.B
		if !adExFinite(spikeW) {
			return 0, ErrAdExNonFiniteUpdate
		}
		s.V = s.VReset
		s.W = spikeW
		return 1, nil
	}
	s.V = nextV
	s.W = nextW
	return 0, nil
}

// Reset restores the dynamic state while preserving configured parameters.
func (s *AdExNeuronState) Reset() {
	s.V = s.VRest
	s.W = 0.0
}

// SimulateComplete returns aligned post-step voltage, adaptation, and event traces.
// The receiver is committed only after every candidate step succeeds.
func (s *AdExNeuronState) SimulateComplete(nSteps int, iExt float64) ([]float64, []float64, []uint8, int, error) {
	if nSteps < 0 {
		return nil, nil, nil, 0, ErrAdExInvalidSteps
	}
	candidate := *s
	vTrace := make([]float64, nSteps)
	wTrace := make([]float64, nSteps)
	events := make([]uint8, nSteps)
	spikes := 0
	for index := range vTrace {
		event, err := candidate.Step(iExt)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		vTrace[index] = candidate.V
		wTrace[index] = candidate.W
		events[index] = uint8(event)
		spikes += event
	}
	s.V = candidate.V
	s.W = candidate.W
	return vTrace, wTrace, events, spikes, nil
}

// SimulateAdExNeuron runs the default neuron for n steps under a constant current.
func SimulateAdExNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAdExNeuron()
	trace, _, _, spikes, err := s.SimulateComplete(nSteps, iExt)
	if err != nil {
		panic(err)
	}
	return trace, spikes
}

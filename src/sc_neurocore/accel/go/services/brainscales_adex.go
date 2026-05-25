// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for brainscales_adex

package services

import (
	"errors"
	"math"
)

var (
	ErrBrainScaleSAdExInvalidInput    = errors.New("brainscales adex input current must be finite")
	ErrBrainScaleSAdExInvalidState    = errors.New("brainscales adex state parameters must be finite with positive delta_t, tau, tau_w, hw_speedup, and dt")
	ErrBrainScaleSAdExNonFiniteUpdate = errors.New("brainscales adex integrator update must remain finite")
)

// BrainScaleSAdExNeuronState holds the neuron state
type BrainScaleSAdExNeuronState struct {
	V          float64
	W          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	DeltaT     float64
	VRh        float64
	Tau        float64
	TauW       float64
	A          float64
	B          float64
	HwSpeedup  float64
	Dt         float64
}

// NewBrainScaleSAdExNeuron creates a new BrainScaleSAdExNeuron neuron with default parameters
func NewBrainScaleSAdExNeuron() *BrainScaleSAdExNeuronState {
	return &BrainScaleSAdExNeuronState{
		V:          -65.0,
		W:          0.0,
		VRest:      -65.0,
		VReset:     -68.0,
		VThreshold: -50.0,
		DeltaT:     2.0,
		VRh:        -55.0,
		Tau:        20.0,
		TauW:       100.0,
		A:          0.5,
		B:          7.0,
		HwSpeedup:  1000.0,
		Dt:         0.1,
	}
}

func brainScaleSAdExFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *BrainScaleSAdExNeuronState) Valid() bool {
	return brainScaleSAdExFinite(s.V, s.W, s.VRest, s.VReset, s.VThreshold, s.DeltaT, s.VRh, s.Tau, s.TauW, s.A, s.B, s.HwSpeedup, s.Dt) &&
		s.DeltaT > 0.0 &&
		s.Tau > 0.0 &&
		s.TauW > 0.0 &&
		s.HwSpeedup > 0.0 &&
		s.Dt > 0.0
}

// Step advances the neuron by one timestep
func (s *BrainScaleSAdExNeuronState) Step(iExt float64) (int, error) {
	if !brainScaleSAdExFinite(iExt) {
		return 0, ErrBrainScaleSAdExInvalidInput
	}
	if !s.Valid() {
		return 0, ErrBrainScaleSAdExInvalidState
	}

	dtHW := s.Dt * s.HwSpeedup
	dtBio := dtHW / s.HwSpeedup
	arg := (s.V - s.VRh) / s.DeltaT
	if arg < -20.0 {
		arg = -20.0
	} else if arg > 20.0 {
		arg = 20.0
	}
	expTerm := s.DeltaT * math.Exp(arg)
	dv := (-(s.V - s.VRest) + expTerm - s.W + iExt) / s.Tau * dtBio
	dw := (s.A*(s.V-s.VRest) - s.W) / s.TauW * dtBio
	nextV := s.V + dv
	nextW := s.W + dw
	if !brainScaleSAdExFinite(dtHW, dtBio, expTerm, dv, dw, nextV, nextW) {
		return 0, ErrBrainScaleSAdExNonFiniteUpdate
	}
	if nextV >= s.VThreshold {
		spikeW := nextW + s.B
		if !brainScaleSAdExFinite(spikeW) {
			return 0, ErrBrainScaleSAdExNonFiniteUpdate
		}
		s.V = s.VReset
		s.W = spikeW
		return 1, nil
	}
	s.V = nextV
	s.W = nextW
	return 0, nil
}

// SimulateBrainScaleSAdExNeuron runs the neuron for n steps
func SimulateBrainScaleSAdExNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBrainScaleSAdExNeuron()
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

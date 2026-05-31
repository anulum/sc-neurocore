// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_nagumo

package services

import "math"

// FitzHughNagumoNeuronState holds the neuron state.
type FitzHughNagumoNeuronState struct {
	V          float64
	W          float64
	A          float64
	B          float64
	Epsilon    float64
	Dt         float64
	VThreshold float64
}

// NewFitzHughNagumoNeuron creates a new FitzHughNagumoNeuron neuron with default parameters.
func NewFitzHughNagumoNeuron() *FitzHughNagumoNeuronState {
	return &FitzHughNagumoNeuronState{V: -1.0, W: -0.5, A: 0.7, B: 0.8, Epsilon: 0.08, Dt: 0.1, VThreshold: 1.0}
}

// Step advances the neuron by one RK4 timestep. It returns -1 without mutating on invalid input.
func (s *FitzHughNagumoNeuronState) Step(iExt float64) int {
	if !finiteFitzHughNagumo(iExt) || !ValidateFitzHughNagumoNeuron(s) {
		return -1
	}
	vPrev := s.V
	newV, newW, ok := rk4FitzHughNagumoCandidate(s, iExt)
	if !ok || !finiteFitzHughNagumo(newV, newW) {
		return -1
	}
	s.V = newV
	s.W = newW
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateFitzHughNagumoNeuron runs the neuron for n steps.
func SimulateFitzHughNagumoNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughNagumoNeuron()
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

func rhsFitzHughNagumo(s *FitzHughNagumoNeuronState, v, w, iExt float64) (float64, float64, bool) {
	if !finiteFitzHughNagumo(v, w, iExt) {
		return 0, 0, false
	}
	dv := v - math.Pow(v, 3.0)/3.0 - w + iExt
	dw := s.Epsilon * (v + s.A - s.B*w)
	return dv, dw, finiteFitzHughNagumo(dv, dw)
}

func rk4FitzHughNagumoCandidate(s *FitzHughNagumoNeuronState, iExt float64) (float64, float64, bool) {
	v0, w0, dt := s.V, s.W, s.Dt
	k1v, k1w, ok := rhsFitzHughNagumo(s, v0, w0, iExt)
	if !ok {
		return 0, 0, false
	}
	k2v, k2w, ok := rhsFitzHughNagumo(s, v0+0.5*dt*k1v, w0+0.5*dt*k1w, iExt)
	if !ok {
		return 0, 0, false
	}
	k3v, k3w, ok := rhsFitzHughNagumo(s, v0+0.5*dt*k2v, w0+0.5*dt*k2w, iExt)
	if !ok {
		return 0, 0, false
	}
	k4v, k4w, ok := rhsFitzHughNagumo(s, v0+dt*k3v, w0+dt*k3w, iExt)
	if !ok {
		return 0, 0, false
	}
	return v0 + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0, w0 + dt*(k1w+2.0*k2w+2.0*k3w+k4w)/6.0, true
}

func finiteFitzHughNagumo(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func ValidateFitzHughNagumoNeuron(s *FitzHughNagumoNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteFitzHughNagumo(s.V, s.W, s.A, s.B, s.Epsilon, s.Dt, s.VThreshold) && s.B > 0.0 && s.Epsilon > 0.0 && s.Dt > 0.0
}

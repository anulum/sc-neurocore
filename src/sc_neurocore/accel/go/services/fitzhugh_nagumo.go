// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_nagumo

package services

import (
	"errors"
	"math"
)

var (
	// ErrFitzHughNagumoInvalidInput reports a non-finite input current.
	ErrFitzHughNagumoInvalidInput = errors.New("fitzhugh-nagumo input current must be finite")
	// ErrFitzHughNagumoInvalidState reports non-finite state or non-positive b, epsilon, or dt.
	ErrFitzHughNagumoInvalidState = errors.New(
		"fitzhugh-nagumo state parameters must be finite with positive b, epsilon, and dt",
	)
	// ErrFitzHughNagumoNonFiniteUpdate reports an RK4 candidate that left the finite range.
	ErrFitzHughNagumoNonFiniteUpdate = errors.New("fitzhugh-nagumo integrator update must remain finite")
)

// FitzHughNagumoNeuronState holds the two-state FitzHugh-Nagumo (1961) excitable
// system integrated with RK4, in parity with
// sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.
type FitzHughNagumoNeuronState struct {
	V          float64
	W          float64
	A          float64
	B          float64
	Epsilon    float64
	Dt         float64
	VThreshold float64
}

// NewFitzHughNagumoNeuron creates a FitzHugh-Nagumo neuron with the published defaults.
func NewFitzHughNagumoNeuron() *FitzHughNagumoNeuronState {
	return &FitzHughNagumoNeuronState{
		V: -1.0, W: -0.5, A: 0.7, B: 0.8, Epsilon: 0.08, Dt: 0.1, VThreshold: 1.0,
	}
}

func fhnFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Valid reports whether the state and parameters are finite with strictly
// positive b, epsilon, and dt.
func (s *FitzHughNagumoNeuronState) Valid() bool {
	return fhnFinite(s.V, s.W, s.A, s.B, s.Epsilon, s.Dt, s.VThreshold) &&
		s.B > 0.0 && s.Epsilon > 0.0 && s.Dt > 0.0
}

// rhs evaluates the FitzHugh-Nagumo derivatives at (v, w) under a constant input.
// The cube is written v*v*v (exact IEEE multiplication, no transcendental
// functions) so the trace is bit-identical to the Rust powi(3) / Julia v^3 /
// Go-cgo / Mojo v*v*v polyglot simulate backends.
func (s *FitzHughNagumoNeuronState) rhs(v, w, current float64) (float64, float64, bool) {
	dv := v - v*v*v/3.0 - w + current
	dw := s.Epsilon * (v + s.A - s.B*w)
	return dv, dw, fhnFinite(dv, dw)
}

// rk4Candidate returns the classic four-stage Runge-Kutta update for one dt,
// or ok=false if any stage or the combined step left the finite range.
func (s *FitzHughNagumoNeuronState) rk4Candidate(current float64) (float64, float64, bool) {
	v0, w0, dt := s.V, s.W, s.Dt
	k1v, k1w, ok := s.rhs(v0, w0, current)
	if !ok {
		return 0, 0, false
	}
	k2v, k2w, ok := s.rhs(v0+0.5*dt*k1v, w0+0.5*dt*k1w, current)
	if !ok {
		return 0, 0, false
	}
	k3v, k3w, ok := s.rhs(v0+0.5*dt*k2v, w0+0.5*dt*k2w, current)
	if !ok {
		return 0, 0, false
	}
	k4v, k4w, ok := s.rhs(v0+dt*k3v, w0+dt*k3w, current)
	if !ok {
		return 0, 0, false
	}
	nextV := v0 + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextW := w0 + dt*(k1w+2.0*k2w+2.0*k3w+k4w)/6.0
	return nextV, nextW, fhnFinite(nextV, nextW)
}

// Step advances the neuron by one RK4 step under a constant input, matching
// FitzHughNagumoNeuron.step: it returns 1 on an upward crossing of VThreshold
// and 0 otherwise, and fails closed — the state is left unchanged — on a
// non-finite input, state, or candidate.
func (s *FitzHughNagumoNeuronState) Step(iExt float64) (int, error) {
	if !fhnFinite(iExt) {
		return 0, ErrFitzHughNagumoInvalidInput
	}
	if !s.Valid() {
		return 0, ErrFitzHughNagumoInvalidState
	}
	vPrev := s.V
	nextV, nextW, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0, ErrFitzHughNagumoNonFiniteUpdate
	}
	s.V = nextV
	s.W = nextW
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateFitzHughNagumoNeuron runs a default neuron for nSteps RK4 steps under a
// constant current, returning the v trace and the upward-crossing spike count.
// It reproduces FitzHughNagumoNeuron(...).simulate(nSteps, iExt) (the RK4 path).
// On a non-finite update the step fails closed and the last valid v is recorded.
func SimulateFitzHughNagumoNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughNagumoNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		trace[t] = s.V
		if err != nil {
			continue
		}
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

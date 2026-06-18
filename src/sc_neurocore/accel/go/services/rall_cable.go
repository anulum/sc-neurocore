// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for rall_cable

package services

import (
	"math"
)

// RallCableNeuronState holds the neuron state
type RallCableNeuronState struct {
	NComp      int
	TauM       float64
	VRest      float64
	GRatio     float64
	VThreshold float64
	VReset     float64
	Dt         float64
	V          []float64
}

// NewRallCableNeuron creates a new RallCableNeuron neuron with default parameters
func NewRallCableNeuron() *RallCableNeuronState {
	return NewRallCableNeuronWithCompartments(5)
}

// NewRallCableNeuronWithCompartments creates a cable with n compartments.
func NewRallCableNeuronWithCompartments(nComp int) *RallCableNeuronState {
	if nComp < 1 {
		nComp = 1
	}
	v := make([]float64, nComp)
	for i := range v {
		v[i] = -65.0
	}
	return &RallCableNeuronState{
		NComp:      nComp,
		TauM:       20.0,
		VRest:      -65.0,
		GRatio:     0.5,
		VThreshold: -50.0,
		VReset:     -65.0,
		Dt:         0.1,
		V:          v,
	}
}

func finiteRallCable(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func (s *RallCableNeuronState) valid() bool {
	if s == nil || s.NComp < 1 || len(s.V) != s.NComp {
		return false
	}
	if !finiteRallCable(s.TauM) || s.TauM <= 0.0 ||
		!finiteRallCable(s.VRest) ||
		!finiteRallCable(s.GRatio) || s.GRatio < 0.0 ||
		!finiteRallCable(s.VThreshold) ||
		!finiteRallCable(s.VReset) ||
		!finiteRallCable(s.Dt) || s.Dt <= 0.0 {
		return false
	}
	for _, value := range s.V {
		if !finiteRallCable(value) {
			return false
		}
	}
	return true
}

func solveTridiagonal(lower []float64, diagonal []float64, upper []float64, rhs []float64) ([]float64, bool) {
	n := len(diagonal)
	if n == 0 || len(rhs) != n || len(lower) != n-1 || len(upper) != n-1 {
		return nil, false
	}
	cPrime := make([]float64, max(n-1, 0))
	dPrime := make([]float64, n)
	pivot := diagonal[0]
	if !finiteRallCable(pivot) || pivot == 0.0 {
		return nil, false
	}
	if n > 1 {
		cPrime[0] = upper[0] / pivot
	}
	dPrime[0] = rhs[0] / pivot
	for i := 1; i < n; i++ {
		pivot = diagonal[i] - lower[i-1]*cPrime[i-1]
		if !finiteRallCable(pivot) || pivot == 0.0 {
			return nil, false
		}
		if i < n-1 {
			cPrime[i] = upper[i] / pivot
		}
		dPrime[i] = (rhs[i] - lower[i-1]*dPrime[i-1]) / pivot
	}
	solution := make([]float64, n)
	solution[n-1] = dPrime[n-1]
	for i := n - 2; i >= 0; i-- {
		solution[i] = dPrime[i] - cPrime[i]*solution[i+1]
	}
	for _, value := range solution {
		if !finiteRallCable(value) {
			return nil, false
		}
	}
	return solution, true
}

func (s *RallCableNeuronState) candidate(iExt float64) ([]float64, bool) {
	if !s.valid() || !finiteRallCable(iExt) {
		return nil, false
	}
	alpha := s.Dt / s.TauM
	offdiag := -alpha * s.GRatio
	diagonal := make([]float64, s.NComp)
	for i := range diagonal {
		diagonal[i] = 1.0 + alpha + 2.0*alpha*s.GRatio
	}
	if s.NComp == 1 {
		diagonal[0] = 1.0 + alpha
	} else {
		diagonal[0] = 1.0 + alpha + alpha*s.GRatio
		diagonal[s.NComp-1] = 1.0 + alpha + alpha*s.GRatio
	}
	lower := make([]float64, max(s.NComp-1, 0))
	upper := make([]float64, max(s.NComp-1, 0))
	for i := range lower {
		lower[i] = offdiag
		upper[i] = offdiag
	}
	rhs := make([]float64, s.NComp)
	for i := range rhs {
		rhs[i] = s.V[i] - s.VRest
	}
	rhs[s.NComp-1] += alpha * iExt
	solved, ok := solveTridiagonal(lower, diagonal, upper, rhs)
	if !ok {
		return nil, false
	}
	for i := range solved {
		solved[i] += s.VRest
	}
	return solved, true
}

// Step advances the neuron by one timestep
func (s *RallCableNeuronState) Step(iExt float64) int {
	candidate, ok := s.candidate(iExt)
	if !ok {
		return -1
	}
	vPrev := s.V[0]
	if candidate[0] >= s.VThreshold && vPrev < s.VThreshold {
		candidate[0] = s.VReset
		s.V = candidate
		return 1
	}
	s.V = candidate
	return 0
}

// SimulateRallCableNeuron runs the neuron for n steps
func SimulateRallCableNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewRallCableNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V[0]
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

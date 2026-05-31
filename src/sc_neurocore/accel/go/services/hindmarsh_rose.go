// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hindmarsh_rose

package services

import (
	"math"
)

// HindmarshRoseNeuronState holds the neuron state
type HindmarshRoseNeuronState struct {
	X          float64
	Y          float64
	Z          float64
	B          float64
	R          float64
	S          float64
	XRest      float64
	Dt         float64
	XThreshold float64
}

// NewHindmarshRoseNeuron creates a new HindmarshRoseNeuron neuron with default parameters
func NewHindmarshRoseNeuron() *HindmarshRoseNeuronState {
	return &HindmarshRoseNeuronState{
		X:          -1.6,
		Y:          -10.0,
		Z:          2.0,
		B:          3.0,
		R:          0.001,
		S:          4.0,
		XRest:      -1.6,
		Dt:         0.1,
		XThreshold: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *HindmarshRoseNeuronState) Step(current float64) int {
	if !validateHindmarshRoseState(s) || !finiteHindmarshRose(current) {
		return 0
	}
	xPrev := s.X
	x0, y0, z0 := s.X, s.Y, s.Z
	dt := s.Dt
	k1, ok := s.derivatives(x0, y0, z0, current)
	if !ok {
		return 0
	}
	k2, ok := s.derivatives(x0+0.5*dt*k1[0], y0+0.5*dt*k1[1], z0+0.5*dt*k1[2], current)
	if !ok {
		return 0
	}
	k3, ok := s.derivatives(x0+0.5*dt*k2[0], y0+0.5*dt*k2[1], z0+0.5*dt*k2[2], current)
	if !ok {
		return 0
	}
	k4, ok := s.derivatives(x0+dt*k3[0], y0+dt*k3[1], z0+dt*k3[2], current)
	if !ok {
		return 0
	}
	nextX := x0 + (dt/6.0)*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
	nextY := y0 + (dt/6.0)*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
	nextZ := z0 + (dt/6.0)*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])
	if !(finiteHindmarshRose(nextX) && finiteHindmarshRose(nextY) && finiteHindmarshRose(nextZ)) {
		return 0
	}
	s.X = nextX
	s.Y = nextY
	s.Z = nextZ
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1
	}
	return 0
}

func (s *HindmarshRoseNeuronState) derivatives(x, y, z, current float64) ([3]float64, bool) {
	if !(finiteHindmarshRose(x) && finiteHindmarshRose(y) && finiteHindmarshRose(z) && finiteHindmarshRose(current)) {
		return [3]float64{}, false
	}
	derivative := [3]float64{
		y - x*x*x + s.B*x*x - z + current,
		1.0 - 5.0*x*x - y,
		s.R * (s.S*(x-s.XRest) - z),
	}
	return derivative, finiteHindmarshRose(derivative[0]) &&
		finiteHindmarshRose(derivative[1]) &&
		finiteHindmarshRose(derivative[2])
}

// SimulateHindmarshRoseNeuron runs the neuron for n steps
func SimulateHindmarshRoseNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHindmarshRoseNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

func finiteHindmarshRose(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateHindmarshRoseState(s *HindmarshRoseNeuronState) bool {
	return finiteHindmarshRose(s.X) &&
		finiteHindmarshRose(s.Y) &&
		finiteHindmarshRose(s.Z) &&
		finiteHindmarshRose(s.B) &&
		finiteHindmarshRose(s.R) &&
		finiteHindmarshRose(s.S) &&
		finiteHindmarshRose(s.XRest) &&
		finiteHindmarshRose(s.Dt) &&
		finiteHindmarshRose(s.XThreshold) &&
		s.R > 0.0 &&
		s.S > 0.0 &&
		s.Dt > 0.0
}

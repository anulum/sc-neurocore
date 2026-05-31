// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_rinzel

package services

import "math"

func isFiniteFitzHughRinzel(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

type FitzHughRinzelNeuronState struct {
	V          float64
	W          float64
	Y          float64
	A          float64
	B          float64
	C          float64
	D          float64
	Delta      float64
	Mu         float64
	Dt         float64
	VThreshold float64
}

func NewFitzHughRinzelNeuron() *FitzHughRinzelNeuronState {
	return &FitzHughRinzelNeuronState{V: -1.0, W: -0.5, Y: 0.0, A: 0.7, B: 0.8, C: -0.775, D: 1.0, Delta: 0.08, Mu: 0.0001, Dt: 0.1, VThreshold: 1.0}
}

func (s *FitzHughRinzelNeuronState) valid() bool {
	return isFiniteFitzHughRinzel(s.V) && isFiniteFitzHughRinzel(s.W) && isFiniteFitzHughRinzel(s.Y) &&
		isFiniteFitzHughRinzel(s.A) && isFiniteFitzHughRinzel(s.B) && isFiniteFitzHughRinzel(s.C) && isFiniteFitzHughRinzel(s.D) &&
		isFiniteFitzHughRinzel(s.Delta) && isFiniteFitzHughRinzel(s.Mu) && isFiniteFitzHughRinzel(s.Dt) && isFiniteFitzHughRinzel(s.VThreshold) &&
		s.B > 0.0 && s.D > 0.0 && s.Delta > 0.0 && s.Mu > 0.0 && s.Dt > 0.0
}

func (s *FitzHughRinzelNeuronState) derivatives(v, w, y, iExt float64) (float64, float64, float64, bool) {
	if !(isFiniteFitzHughRinzel(v) && isFiniteFitzHughRinzel(w) && isFiniteFitzHughRinzel(y) && isFiniteFitzHughRinzel(iExt)) {
		return 0, 0, 0, false
	}
	dv := v - v*v*v/3.0 - w + y + iExt
	dw := s.Delta * (s.A + v - s.B*w)
	dy := s.Mu * (s.C - v - s.D*y)
	return dv, dw, dy, isFiniteFitzHughRinzel(dv) && isFiniteFitzHughRinzel(dw) && isFiniteFitzHughRinzel(dy)
}

func (s *FitzHughRinzelNeuronState) rk4Candidate(iExt float64) (float64, float64, float64, bool) {
	v0, w0, y0, dt := s.V, s.W, s.Y, s.Dt
	k1v, k1w, k1y, ok := s.derivatives(v0, w0, y0, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k2v, k2w, k2y, ok := s.derivatives(v0+0.5*dt*k1v, w0+0.5*dt*k1w, y0+0.5*dt*k1y, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k3v, k3w, k3y, ok := s.derivatives(v0+0.5*dt*k2v, w0+0.5*dt*k2w, y0+0.5*dt*k2y, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k4v, k4w, k4y, ok := s.derivatives(v0+dt*k3v, w0+dt*k3w, y0+dt*k3y, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	return v0 + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0,
		w0 + dt*(k1w+2.0*k2w+2.0*k3w+k4w)/6.0,
		y0 + dt*(k1y+2.0*k2y+2.0*k3y+k4y)/6.0,
		true
}

func (s *FitzHughRinzelNeuronState) Step(iExt float64) int {
	if !s.valid() || !isFiniteFitzHughRinzel(iExt) {
		return 0
	}
	vPrev := s.V
	nextV, nextW, nextY, ok := s.rk4Candidate(iExt)
	if !(ok && isFiniteFitzHughRinzel(nextV) && isFiniteFitzHughRinzel(nextW) && isFiniteFitzHughRinzel(nextY)) {
		return 0
	}
	s.V = nextV
	s.W = nextW
	s.Y = nextY
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

func SimulateFitzHughRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughRinzelNeuron()
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

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Mihalas-Niebur RK4 dynamics

package services

import "math"

// MihalasNieburNeuronState holds the neuron state.
type MihalasNieburNeuronState struct {
	V          float64
	Theta      float64
	I1         float64
	I2         float64
	VRest      float64
	VReset     float64
	ThetaReset float64
	ThetaInf   float64
	TauV       float64
	TauTheta   float64
	Tau1       float64
	Tau2       float64
	A          float64
	B          float64
	R1         float64
	R2         float64
	Dt         float64
}

// NewMihalasNieburNeuron creates a new MihalasNieburNeuron neuron with default parameters.
func NewMihalasNieburNeuron() *MihalasNieburNeuronState {
	return &MihalasNieburNeuronState{
		V: 0.0, Theta: 1.0, I1: 0.0, I2: 0.0, VRest: 0.0, VReset: 0.0,
		ThetaReset: 1.0, ThetaInf: 1.0, TauV: 10.0, TauTheta: 100.0,
		Tau1: 10.0, Tau2: 200.0, A: 0.0, B: 0.0, R1: 0.0, R2: 0.0, Dt: 1.0,
	}
}

func mihalasNieburFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *MihalasNieburNeuronState) validRuntime() bool {
	return mihalasNieburFinite(
		s.V, s.Theta, s.I1, s.I2, s.VRest, s.VReset, s.ThetaReset, s.ThetaInf,
		s.TauV, s.TauTheta, s.Tau1, s.Tau2, s.A, s.B, s.R1, s.R2, s.Dt,
	) && s.TauV > 0.0 && s.TauTheta > 0.0 && s.Tau1 > 0.0 && s.Tau2 > 0.0 && s.Dt > 0.0
}

func (s *MihalasNieburNeuronState) derivatives(v float64, theta float64, i1 float64, i2 float64, iExt float64) [4]float64 {
	return [4]float64{
		(-(v - s.VRest) + i1 + i2 + iExt) / s.TauV,
		(s.ThetaInf - theta + s.A*(v-s.VRest)) / s.TauTheta,
		-i1 / s.Tau1,
		-i2 / s.Tau2,
	}
}

func mihalasNieburAddScaled(state [4]float64, slope [4]float64, scale float64) [4]float64 {
	return [4]float64{
		state[0] + scale*slope[0],
		state[1] + scale*slope[1],
		state[2] + scale*slope[2],
		state[3] + scale*slope[3],
	}
}

func (s *MihalasNieburNeuronState) rk4Candidate(iExt float64) ([4]float64, bool) {
	state := [4]float64{s.V, s.Theta, s.I1, s.I2}
	halfDt := 0.5 * s.Dt
	k1 := s.derivatives(state[0], state[1], state[2], state[3], iExt)
	s2 := mihalasNieburAddScaled(state, k1, halfDt)
	k2 := s.derivatives(s2[0], s2[1], s2[2], s2[3], iExt)
	s3 := mihalasNieburAddScaled(state, k2, halfDt)
	k3 := s.derivatives(s3[0], s3[1], s3[2], s3[3], iExt)
	s4 := mihalasNieburAddScaled(state, k3, s.Dt)
	k4 := s.derivatives(s4[0], s4[1], s4[2], s4[3], iExt)
	candidate := [4]float64{
		state[0] + s.Dt*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0,
		state[1] + s.Dt*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0,
		state[2] + s.Dt*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0,
		state[3] + s.Dt*(k1[3]+2.0*k2[3]+2.0*k3[3]+k4[3])/6.0,
	}
	return candidate, mihalasNieburFinite(candidate[0], candidate[1], candidate[2], candidate[3])
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *MihalasNieburNeuronState) Step(iExt float64) int {
	if !mihalasNieburFinite(iExt) || !s.validRuntime() {
		return 0
	}
	candidate, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V, s.Theta, s.I1, s.I2 = candidate[0], candidate[1], candidate[2], candidate[3]
	if s.V >= s.Theta {
		s.V = s.VReset + s.B*(s.V-s.VRest)
		s.Theta = math.Max(s.Theta, s.ThetaReset)
		s.I1 += s.R1
		s.I2 += s.R2
		return 1
	}
	return 0
}

// SimulateMihalasNieburNeuron runs the neuron for n steps.
func SimulateMihalasNieburNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMihalasNieburNeuron()
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

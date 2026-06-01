// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for GLIF RK4 dynamics

package services

import "math"

// GLIFNeuronState holds the neuron state.
type GLIFNeuronState struct {
	V          float64
	Theta      float64
	ThetaInf   float64
	IAsc1      float64
	IAsc2      float64
	VRest      float64
	VReset     float64
	TauM       float64
	TauTheta   float64
	TauAsc1    float64
	TauAsc2    float64
	ATheta     float64
	DeltaTheta float64
	RAsc1      float64
	RAsc2      float64
	Resistance float64
	Dt         float64
}

// NewGLIFNeuron creates a new GLIFNeuron neuron with default parameters.
func NewGLIFNeuron() *GLIFNeuronState {
	return &GLIFNeuronState{
		V: -70.0, Theta: -50.0, ThetaInf: -50.0, IAsc1: 0.0, IAsc2: 0.0,
		VRest: -70.0, VReset: -70.0, TauM: 10.0, TauTheta: 100.0,
		TauAsc1: 10.0, TauAsc2: 200.0, ATheta: 0.01, DeltaTheta: 2.0,
		RAsc1: 1.0, RAsc2: 0.5, Resistance: 1.0, Dt: 1.0,
	}
}

func glifFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *GLIFNeuronState) validRuntime() bool {
	return glifFinite(
		s.V, s.Theta, s.ThetaInf, s.IAsc1, s.IAsc2, s.VRest, s.VReset,
		s.TauM, s.TauTheta, s.TauAsc1, s.TauAsc2, s.ATheta, s.DeltaTheta,
		s.RAsc1, s.RAsc2, s.Resistance, s.Dt,
	) && s.TauM > 0.0 && s.TauTheta > 0.0 && s.TauAsc1 > 0.0 && s.TauAsc2 > 0.0 &&
		s.Dt > 0.0 && s.DeltaTheta >= 0.0 && s.Resistance >= 0.0
}

func (s *GLIFNeuronState) derivatives(v float64, theta float64, iAsc1 float64, iAsc2 float64, iExt float64) [4]float64 {
	return [4]float64{
		(-(v - s.VRest) + s.Resistance*iExt + iAsc1 + iAsc2) / s.TauM,
		(s.ThetaInf - theta + s.ATheta*(v-s.VRest)) / s.TauTheta,
		-iAsc1 / s.TauAsc1,
		-iAsc2 / s.TauAsc2,
	}
}

func glifAddScaled(state [4]float64, slope [4]float64, scale float64) [4]float64 {
	return [4]float64{
		state[0] + scale*slope[0],
		state[1] + scale*slope[1],
		state[2] + scale*slope[2],
		state[3] + scale*slope[3],
	}
}

func (s *GLIFNeuronState) rk4Candidate(iExt float64) ([4]float64, bool) {
	state := [4]float64{s.V, s.Theta, s.IAsc1, s.IAsc2}
	halfDt := 0.5 * s.Dt
	k1 := s.derivatives(state[0], state[1], state[2], state[3], iExt)
	s2 := glifAddScaled(state, k1, halfDt)
	k2 := s.derivatives(s2[0], s2[1], s2[2], s2[3], iExt)
	s3 := glifAddScaled(state, k2, halfDt)
	k3 := s.derivatives(s3[0], s3[1], s3[2], s3[3], iExt)
	s4 := glifAddScaled(state, k3, s.Dt)
	k4 := s.derivatives(s4[0], s4[1], s4[2], s4[3], iExt)
	candidate := [4]float64{
		state[0] + s.Dt*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0,
		state[1] + s.Dt*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0,
		state[2] + s.Dt*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0,
		state[3] + s.Dt*(k1[3]+2.0*k2[3]+2.0*k3[3]+k4[3])/6.0,
	}
	return candidate, glifFinite(candidate[0], candidate[1], candidate[2], candidate[3])
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *GLIFNeuronState) Step(iExt float64) int {
	if !glifFinite(iExt) || !s.validRuntime() {
		return 0
	}
	candidate, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V, s.Theta, s.IAsc1, s.IAsc2 = candidate[0], candidate[1], candidate[2], candidate[3]
	if s.V >= s.Theta {
		s.V = s.VReset
		s.Theta += s.DeltaTheta
		s.IAsc1 += s.RAsc1
		s.IAsc2 += s.RAsc2
		return 1
	}
	return 0
}

// SimulateGLIFNeuron runs the neuron for n steps.
func SimulateGLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGLIFNeuron()
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

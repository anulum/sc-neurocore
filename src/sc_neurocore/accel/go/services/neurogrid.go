// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for neurogrid

package services

import "math"

// NeuroGridNeuronState holds the reduced Neurogrid two-compartment state.
type NeuroGridNeuronState struct {
	VS         float64
	VD         float64
	TauS       float64
	TauD       float64
	GC         float64
	DeltaT     float64
	VRest      float64
	VThreshold float64
	VPeak      float64
	VReset     float64
	Dt         float64
}

// NewNeuroGridNeuron creates a new NeuroGridNeuron neuron with default parameters.
func NewNeuroGridNeuron() *NeuroGridNeuronState {
	return &NeuroGridNeuronState{
		VS:         -65.0,
		VD:         -65.0,
		TauS:       20.0,
		TauD:       50.0,
		GC:         0.5,
		DeltaT:     2.0,
		VRest:      -65.0,
		VThreshold: -50.0,
		VPeak:      20.0,
		VReset:     -65.0,
		Dt:         0.1,
	}
}

func neurogridFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *NeuroGridNeuronState) valid() bool {
	return neurogridFinite(s.VS, s.VD, s.TauS, s.TauD, s.GC, s.DeltaT, s.VRest, s.VThreshold, s.VPeak, s.VReset, s.Dt) &&
		s.TauS > 0.0 && s.TauD > 0.0 && s.DeltaT > 0.0 && s.Dt > 0.0 && s.GC >= 0.0
}

func (s *NeuroGridNeuronState) derivatives(vs float64, vd float64, current float64) (float64, float64) {
	vsEff := math.Min(vs, s.VPeak)
	dVD := (-(vd - s.VRest) + current - s.GC*(vd-vsEff)) / s.TauD
	expArg := math.Min((vsEff-s.VThreshold)/s.DeltaT, 20.0)
	expTerm := s.DeltaT * math.Exp(expArg)
	dVS := (-(vsEff - s.VRest) + expTerm + s.GC*(vd-vsEff)) / s.TauS
	return dVS, dVD
}

func (s *NeuroGridNeuronState) rk4Substep(vs float64, vd float64, current float64) (float64, float64) {
	dt := s.Dt
	k1vs, k1vd := s.derivatives(vs, vd, current)
	k2vs, k2vd := s.derivatives(vs+0.5*dt*k1vs, vd+0.5*dt*k1vd, current)
	k3vs, k3vd := s.derivatives(vs+0.5*dt*k2vs, vd+0.5*dt*k2vd, current)
	k4vs, k4vd := s.derivatives(vs+dt*k3vs, vd+dt*k3vd, current)
	nextVS := vs + dt*(k1vs+2.0*k2vs+2.0*k3vs+k4vs)/6.0
	nextVD := vd + dt*(k1vd+2.0*k2vd+2.0*k3vd+k4vd)/6.0
	return nextVS, nextVD
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *NeuroGridNeuronState) Step(iExt float64) int {
	if !neurogridFinite(iExt) || !s.valid() {
		return 0
	}
	nextVS, nextVD := s.rk4Substep(s.VS, s.VD, iExt)
	if !neurogridFinite(nextVS, nextVD) {
		return 0
	}
	s.VD = nextVD
	if nextVS >= s.VPeak {
		s.VS = s.VReset
		return 1
	}
	s.VS = nextVS
	return 0
}

// SimulateNeuroGridNeuron runs the neuron for n steps.
func SimulateNeuroGridNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewNeuroGridNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VS
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

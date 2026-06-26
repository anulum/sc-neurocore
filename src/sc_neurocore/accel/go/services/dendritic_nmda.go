// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dendritic_nmda

package services

import (
	"math"
)

// DendriticNMDANeuronState holds the two-compartment NMDA neuron state.
type DendriticNMDANeuronState struct {
	GNmda     float64
	ENmda     float64
	MgConc    float64
	GCoupling float64
	TauSoma   float64
	TauDend   float64
	Theta     float64
	Dt        float64
	VSoma     float64
	VDend     float64
}

// NewDendriticNMDANeuron creates a neuron with the Python reference defaults.
func NewDendriticNMDANeuron() *DendriticNMDANeuronState {
	return &DendriticNMDANeuronState{
		GNmda:     1.5,
		ENmda:     0.0,
		MgConc:    1.0,
		GCoupling: 0.5,
		TauSoma:   20.0,
		TauDend:   50.0,
		Theta:     -50.0,
		Dt:        0.1,
		VSoma:     -65.0,
		VDend:     -65.0,
	}
}

func (s *DendriticNMDANeuronState) valid() bool {
	return finite(s.GNmda, s.ENmda, s.MgConc, s.GCoupling, s.TauSoma, s.TauDend, s.Theta, s.Dt, s.VSoma, s.VDend) &&
		s.GNmda >= 0.0 && s.MgConc >= 0.0 && s.GCoupling >= 0.0 &&
		s.TauSoma > 0.0 && s.TauDend > 0.0 && s.Dt > 0.0
}

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *DendriticNMDANeuronState) mgBlock(v float64) float64 {
	return 1.0 / (1.0 + (s.MgConc/3.57)*math.Exp(-0.062*v))
}

func (s *DendriticNMDANeuronState) derivatives(vSoma float64, vDend float64, iSoma float64, glutamate float64) (float64, float64) {
	block := s.mgBlock(vDend)
	iNmda := s.GNmda * glutamate * block * (vDend - s.ENmda)
	dvSoma := (-vSoma - 65.0 + iSoma + s.GCoupling*(vDend-vSoma)) / s.TauSoma
	dvDend := (-vDend - 65.0 + iNmda + s.GCoupling*(vSoma-vDend)) / s.TauDend
	return dvSoma, dvDend
}

func (s *DendriticNMDANeuronState) rk4Substep(vSoma float64, vDend float64, iSoma float64, glutamate float64) (float64, float64) {
	dt := s.Dt
	k1s, k1d := s.derivatives(vSoma, vDend, iSoma, glutamate)
	k2s, k2d := s.derivatives(vSoma+0.5*dt*k1s, vDend+0.5*dt*k1d, iSoma, glutamate)
	k3s, k3d := s.derivatives(vSoma+0.5*dt*k2s, vDend+0.5*dt*k2d, iSoma, glutamate)
	k4s, k4d := s.derivatives(vSoma+dt*k3s, vDend+dt*k3d, iSoma, glutamate)
	return vSoma + dt*(k1s+2.0*k2s+2.0*k3s+k4s)/6.0,
		vDend + dt*(k1d+2.0*k2d+2.0*k3d+k4d)/6.0
}

// Step advances the neuron by one timestep.
func (s *DendriticNMDANeuronState) Step(iSoma float64, glutamateInput ...float64) int {
	glutamate := 0.0
	if len(glutamateInput) > 0 {
		glutamate = glutamateInput[0]
	}
	if !finite(iSoma, glutamate) || glutamate < 0.0 || !s.valid() {
		return 0
	}
	nextVSoma, nextVDend := s.rk4Substep(s.VSoma, s.VDend, iSoma, glutamate)
	if !finite(nextVSoma, nextVDend) {
		return 0
	}
	s.VDend = nextVDend
	if nextVSoma >= s.Theta {
		s.VSoma = -65.0
		return 1
	}
	s.VSoma = nextVSoma
	return 0
}

// SimulateDendriticNMDANeuron runs the neuron for n steps.
func SimulateDendriticNMDANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDendriticNMDANeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VSoma
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

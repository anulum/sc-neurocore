// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for multicompartment_mcn

package services

import (
	"math"
)

// MulticompartmentMCNNeuronState holds the neuron state.
type MulticompartmentMCNNeuronState struct {
	Tau     float64
	TauB    float64
	TauA    float64
	GRatio  float64
	Beta    float64
	VTh     float64
	Dt      float64
	U       float64
	VBasal  float64
	VApical float64
}

// NewMulticompartmentMCNNeuron creates a new MulticompartmentMCNNeuron neuron with default parameters
func NewMulticompartmentMCNNeuron() *MulticompartmentMCNNeuronState {
	return &MulticompartmentMCNNeuronState{
		Tau:     2.0,
		TauB:    2.0,
		TauA:    2.0,
		GRatio:  1.0,
		Beta:    1.0,
		VTh:     1.0,
		Dt:      1.0,
		U:       0.0,
		VBasal:  0.0,
		VApical: 0.0,
	}
}

func mcnFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Sigma returns the apical sigmoid gate sigma(x) = 1/(1 + exp(-beta*x)).
func (s *MulticompartmentMCNNeuronState) Sigma(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-s.Beta*x))
}

func (s *MulticompartmentMCNNeuronState) valid() bool {
	return mcnFinite(s.Tau, s.TauB, s.TauA, s.GRatio, s.Beta, s.VTh, s.Dt, s.U, s.VBasal, s.VApical) &&
		s.Tau > 0.0 && s.TauB > 0.0 && s.TauA > 0.0 && s.Beta > 0.0 && s.VTh > 0.0 &&
		s.Dt > 0.0 && s.GRatio >= 0.0
}

// derivatives returns (dU, dVBasal, dVApical) from one consistent state.
func (s *MulticompartmentMCNNeuronState) derivatives(u, vBasal, vApical, xBasal, xApical, iSoma float64) (float64, float64, float64) {
	gate := s.Sigma(vApical)
	du := (-u + gate*(s.GRatio*(vBasal-u)+iSoma)) / s.Tau
	dvBasal := (-vBasal + xBasal) / s.TauB
	dvApical := (-vApical + xApical) / s.TauA
	return du, dvBasal, dvApical
}

// rk4Substep returns one classical RK4 increment of the three-state vector over Dt.
func (s *MulticompartmentMCNNeuronState) rk4Substep(u, vBasal, vApical, xBasal, xApical, iSoma float64) (float64, float64, float64) {
	dt := s.Dt
	k1u, k1vb, k1va := s.derivatives(u, vBasal, vApical, xBasal, xApical, iSoma)
	k2u, k2vb, k2va := s.derivatives(u+0.5*dt*k1u, vBasal+0.5*dt*k1vb, vApical+0.5*dt*k1va, xBasal, xApical, iSoma)
	k3u, k3vb, k3va := s.derivatives(u+0.5*dt*k2u, vBasal+0.5*dt*k2vb, vApical+0.5*dt*k2va, xBasal, xApical, iSoma)
	k4u, k4vb, k4va := s.derivatives(u+dt*k3u, vBasal+dt*k3vb, vApical+dt*k3va, xBasal, xApical, iSoma)
	nextU := u + dt*(k1u+2.0*k2u+2.0*k3u+k4u)/6.0
	nextVBasal := vBasal + dt*(k1vb+2.0*k2vb+2.0*k3vb+k4vb)/6.0
	nextVApical := vApical + dt*(k1va+2.0*k2va+2.0*k3va+k4va)/6.0
	return nextU, nextVBasal, nextVApical
}

// StepCompartments advances one RK4 step with basal, apical, and somatic drives.
func (s *MulticompartmentMCNNeuronState) StepCompartments(xBasal, xApical, iSoma float64) int {
	if !mcnFinite(xBasal, xApical, iSoma) || !s.valid() {
		return 0
	}
	nextU, nextVBasal, nextVApical := s.rk4Substep(s.U, s.VBasal, s.VApical, xBasal, xApical, iSoma)
	if !mcnFinite(nextU, nextVBasal, nextVApical) {
		return 0
	}
	spike := 0
	if nextU >= s.VTh {
		spike = 1
		nextU = 0.0
	}
	s.U = nextU
	s.VBasal = nextVBasal
	s.VApical = nextVApical
	return spike
}

// Step advances the neuron by one timestep with input to the basal dendrite.
func (s *MulticompartmentMCNNeuronState) Step(iExt float64) int {
	return s.StepCompartments(iExt, 0.0, 0.0)
}

// SimulateMulticompartmentMCNNeuron runs the neuron for n steps.
func SimulateMulticompartmentMCNNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMulticompartmentMCNNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.U
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

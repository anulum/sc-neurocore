// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte et al. 2000 pyramidal-cell service

// Package services contains native neuron service kernels.
package services

import (
	"errors"
	"math"
)

const compteGateMax = 1.0e6

// CompteWMNeuronState owns complete membrane, channel, refractory, and
// source-control-set configuration. Conductances are microSiemens, voltage mV,
// current nA, capacitance nF, and time ms.
type CompteWMNeuronState struct {
	V, SAmpa, SNmda, XNmda, SGaba, RefRemaining float64
	GL, GAmpa, GNmda, GGaba                     float64
	EL, EExc, EInh, CM, Mg                      float64
	TauAmpa, TauNmda, TauX, TauGaba, AlphaNmda  float64
	VThreshold, VReset, TauRef, Dt              float64
}

// NewCompteWMNeuron constructs the Compte (2000) control-set pyramidal cell.
func NewCompteWMNeuron() *CompteWMNeuronState {
	return &CompteWMNeuronState{
		V: -70, GL: 0.025, GAmpa: 0.0031, GNmda: 0.000381, GGaba: 0.001336,
		EL: -70, EExc: 0, EInh: -70, CM: 0.5, Mg: 1,
		TauAmpa: 2, TauNmda: 100, TauX: 2, TauGaba: 10, AlphaNmda: 0.5,
		VThreshold: -50, VReset: -60, TauRef: 2, Dt: 0.02,
	}
}

func finiteCompte(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// ValidateCompteWM checks every mutable state and configuration invariant.
func ValidateCompteWM(s *CompteWMNeuronState) bool {
	if s == nil {
		return false
	}
	values := []float64{
		s.V, s.SAmpa, s.SNmda, s.XNmda, s.SGaba, s.RefRemaining,
		s.GL, s.GAmpa, s.GNmda, s.GGaba, s.EL, s.EExc, s.EInh, s.CM, s.Mg,
		s.TauAmpa, s.TauNmda, s.TauX, s.TauGaba, s.AlphaNmda,
		s.VThreshold, s.VReset, s.TauRef, s.Dt,
	}
	for _, value := range values {
		if !finiteCompte(value) {
			return false
		}
	}
	return s.V >= -200 && s.V <= 100 && s.VReset >= -200 && s.VReset <= 100 &&
		s.SAmpa >= 0 && s.SAmpa <= compteGateMax &&
		s.SNmda >= 0 && s.SNmda <= 1 &&
		s.XNmda >= 0 && s.XNmda <= compteGateMax &&
		s.SGaba >= 0 && s.SGaba <= compteGateMax && s.RefRemaining >= 0 &&
		s.GL >= 0 && s.GAmpa >= 0 && s.GNmda >= 0 && s.GGaba >= 0 &&
		s.Mg >= 0 && s.AlphaNmda >= 0 && s.CM > 0 && s.TauAmpa > 0 &&
		s.TauNmda > 0 && s.TauX > 0 && s.TauGaba > 0 && s.TauRef > 0 && s.Dt > 0
}

func (s *CompteWMNeuronState) derivatives(
	state [5]float64, current float64, membraneActive bool,
) ([5]float64, bool) {
	v, sAmpa, sNmda, xNmda, sGaba := state[0], state[1], state[2], state[3], state[4]
	result := [5]float64{
		0,
		-sAmpa / s.TauAmpa,
		-sNmda/s.TauNmda + s.AlphaNmda*xNmda*(1-sNmda),
		-xNmda / s.TauX,
		-sGaba / s.TauGaba,
	}
	if membraneActive {
		block := 1 / (1 + s.Mg/3.57*math.Exp(-0.062*v))
		iL := s.GL * (v - s.EL)
		iAmpa := s.GAmpa * sAmpa * (v - s.EExc)
		iNmda := s.GNmda * block * sNmda * (v - s.EExc)
		iGaba := s.GGaba * sGaba * (v - s.EInh)
		result[0] = (-iL - iAmpa - iNmda - iGaba + current) / s.CM
	}
	for _, value := range result {
		if !finiteCompte(value) {
			return result, false
		}
	}
	return result, true
}

// StepWithEvents applies separate recurrent-NMDA, external-AMPA, and
// inhibitory-GABAA events, then advances one atomic midpoint-RK2 step.
func (s *CompteWMNeuronState) StepWithEvents(
	current float64, recurrentEvent, externalEvent, inhibitoryEvent bool,
) (int, error) {
	if !ValidateCompteWM(s) || !finiteCompte(current) {
		return 0, errors.New("invalid Compte state, configuration, or current")
	}
	initial := [5]float64{s.V, s.SAmpa, s.SNmda, s.XNmda, s.SGaba}
	if externalEvent {
		initial[1]++
	}
	if recurrentEvent {
		initial[3]++
	}
	if inhibitoryEvent {
		initial[4]++
	}
	for _, value := range initial[1:] {
		if !finiteCompte(value) || value < 0 || value > compteGateMax {
			return 0, errors.New("Compte event candidate outside gate envelope")
		}
	}
	active := s.RefRemaining <= 0
	k1, ok := s.derivatives(initial, current, active)
	if !ok {
		return 0, errors.New("non-finite Compte RK2 first stage")
	}
	var midpoint [5]float64
	for index := range midpoint {
		midpoint[index] = initial[index] + 0.5*s.Dt*k1[index]
	}
	k2, ok := s.derivatives(midpoint, current, active)
	if !ok {
		return 0, errors.New("non-finite Compte RK2 midpoint stage")
	}
	var candidate [5]float64
	for index := range candidate {
		candidate[index] = initial[index] + s.Dt*k2[index]
		if !finiteCompte(candidate[index]) {
			return 0, errors.New("non-finite Compte RK2 candidate")
		}
	}
	if candidate[0] < -200 || candidate[0] > 100 || candidate[2] > 1 {
		return 0, errors.New("Compte RK2 candidate outside safety envelope")
	}
	for _, value := range candidate[1:] {
		if value < 0 || value > compteGateMax {
			return 0, errors.New("Compte gate candidate outside safety envelope")
		}
	}
	event := 0
	refRemaining := math.Max(0, s.RefRemaining-s.Dt)
	if !active {
		candidate[0] = s.VReset
	} else if candidate[0] >= s.VThreshold {
		candidate[0], refRemaining, event = s.VReset, s.TauRef, 1
	}
	s.V, s.SAmpa, s.SNmda, s.XNmda, s.SGaba =
		candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]
	s.RefRemaining = refRemaining
	return event, nil
}

// Step advances the scalar-current compatibility path without presynaptic events.
func (s *CompteWMNeuronState) Step(current float64) (int, error) {
	return s.StepWithEvents(current, false, false, false)
}

// StepWithSpike retains spikeIn as the recurrent NMDA compatibility event.
func (s *CompteWMNeuronState) StepWithSpike(current float64, spikeIn bool) (int, error) {
	return s.StepWithEvents(current, spikeIn, false, false)
}

// Reset clears all dynamic state while preserving configuration.
func (s *CompteWMNeuronState) Reset() {
	s.V, s.SAmpa, s.SNmda, s.XNmda, s.SGaba, s.RefRemaining = s.EL, 0, 0, 0, 0, 0
}

// GetState returns complete dynamic state in public trace order.
func (s *CompteWMNeuronState) GetState() [6]float64 {
	return [6]float64{s.V, s.SAmpa, s.SNmda, s.XNmda, s.SGaba, s.RefRemaining}
}

// SimulateCompteWMNeuron retains the scalar-current service compatibility path.
func SimulateCompteWMNeuron(nSteps int, current float64) ([]float64, int) {
	state := NewCompteWMNeuron()
	trace, spikes := make([]float64, nSteps), 0
	for index := range trace {
		event, err := state.Step(current)
		if err != nil {
			trace[index] = math.NaN()
			continue
		}
		trace[index], spikes = state.V, spikes+event
	}
	return trace, spikes
}

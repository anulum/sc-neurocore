// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for alpha

package services

import (
	"errors"
	"math"
)

// AlphaNeuronState holds the neuron state
type AlphaNeuronState struct {
	V          float64
	AExc       float64
	IExc       float64
	AInh       float64
	IInh       float64
	VRest      float64
	VThreshold float64
	TauV       float64
	TauExc     float64
	TauInh     float64
	Dt         float64
}

// NewAlphaNeuron creates a new AlphaNeuron neuron with default parameters
func NewAlphaNeuron() *AlphaNeuronState {
	return &AlphaNeuronState{
		V:          0.0,
		AExc:       0.0,
		IExc:       0.0,
		AInh:       0.0,
		IInh:       0.0,
		VRest:      0.0,
		VThreshold: 1.0,
		TauV:       20.0,
		TauExc:     5.0,
		TauInh:     10.0,
		Dt:         1.0,
	}
}

// Step advances the neuron by one timestep
func (s *AlphaNeuronState) Step(excCurrent float64, inhCurrentValues ...float64) (int, error) {
	inhCurrent := 0.0
	if len(inhCurrentValues) > 1 {
		return 0, errors.New("alpha step accepts at most one inhibitory current")
	}
	if len(inhCurrentValues) == 1 {
		inhCurrent = inhCurrentValues[0]
	}
	if !alphaFinite(excCurrent) || !alphaFinite(inhCurrent) {
		return 0, errors.New("alpha currents must be finite")
	}
	if err := validateAlpha(s); err != nil {
		return 0, err
	}

	excSteady := s.TauExc * excCurrent
	inhSteady := s.TauInh * inhCurrent
	excRiseDelta := s.AExc - excSteady
	inhRiseDelta := s.AInh - inhSteady
	excCurrentDelta := s.IExc - excSteady
	inhCurrentDelta := s.IInh - inhSteady

	aExcNext, iExcNext := alphaFilterCandidates(s.AExc, s.IExc, excCurrent, s.TauExc, s.Dt)
	aInhNext, iInhNext := alphaFilterCandidates(s.AInh, s.IInh, inhCurrent, s.TauInh, s.Dt)
	vSteady := s.VRest + excSteady - inhSteady
	vNext := vSteady +
		(s.V-vSteady)*math.Exp(-s.Dt/s.TauV) +
		alphaMembraneDriveContribution(excCurrentDelta, excRiseDelta, s.TauExc, s.TauV, s.Dt) -
		alphaMembraneDriveContribution(inhCurrentDelta, inhRiseDelta, s.TauInh, s.TauV, s.Dt)
	if !alphaFinite(aExcNext) || !alphaFinite(iExcNext) ||
		!alphaFinite(aInhNext) || !alphaFinite(iInhNext) || !alphaFinite(vNext) {
		return 0, errors.New("alpha exact-flow update became non-finite")
	}

	s.AExc = aExcNext
	s.IExc = iExcNext
	s.AInh = aInhNext
	s.IInh = iInhNext
	if vNext >= s.VThreshold {
		s.V = s.VRest
		return 1, nil
	}
	s.V = vNext
	return 0, nil
}

// SimulateAlphaNeuron runs the neuron for n steps
func SimulateAlphaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAlphaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			break
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

func alphaFilterCandidates(
	riseState float64,
	currentState float64,
	drive float64,
	tau float64,
	dt float64,
) (float64, float64) {
	steadyState := tau * drive
	riseDelta := riseState - steadyState
	currentDelta := currentState - steadyState
	decay := math.Exp(-dt / tau)
	riseNext := steadyState + riseDelta*decay
	currentNext := steadyState + decay*(currentDelta+riseDelta*dt/tau)
	return riseNext, currentNext
}

func alphaMembraneDriveContribution(
	currentDelta float64,
	riseDelta float64,
	tauDrive float64,
	tauV float64,
	dt float64,
) float64 {
	rateV := 1.0 / tauV
	rateDrive := 1.0 / tauDrive
	decayV := math.Exp(-dt / tauV)
	decayDrive := math.Exp(-dt / tauDrive)
	if math.Abs(rateV-rateDrive) <= 1.0e-14 {
		return rateV * decayV * (currentDelta*dt + riseDelta*dt*dt/(2.0*tauDrive))
	}
	rateDelta := rateV - rateDrive
	firstOrder := currentDelta * (decayDrive - decayV) / rateDelta
	secondOrder := riseDelta / tauDrive *
		(decayDrive*(rateDelta*dt-1.0) + decayV) / (rateDelta * rateDelta)
	return rateV * (firstOrder + secondOrder)
}

func validateAlpha(s *AlphaNeuronState) error {
	if s == nil {
		return errors.New("alpha state must not be nil")
	}
	if !alphaFinite(s.V) || !alphaFinite(s.AExc) || !alphaFinite(s.IExc) ||
		!alphaFinite(s.AInh) || !alphaFinite(s.IInh) ||
		!alphaFinite(s.VRest) || !alphaFinite(s.VThreshold) {
		return errors.New("alpha state variables must be finite")
	}
	if !alphaPositive(s.TauV) || !alphaPositive(s.TauExc) ||
		!alphaPositive(s.TauInh) || !alphaPositive(s.Dt) {
		return errors.New("alpha time constants and timestep must be finite and positive")
	}
	return nil
}

func alphaPositive(value float64) bool {
	return alphaFinite(value) && value > 0.0
}

func alphaFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

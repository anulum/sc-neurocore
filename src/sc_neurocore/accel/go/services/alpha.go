// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dual alpha-synapse LIF

package services

import (
	"errors"
	"math"
)

// AlphaNeuronState holds the membrane potential and the two alpha cascades.
type AlphaNeuronState struct {
	V           float64
	AExc        float64
	IExc        float64
	AInh        float64
	IInh        float64
	VRest       float64
	VThreshold  float64
	TauV        float64
	TauExc      float64
	TauInh      float64
	Dt          float64
}

// NewAlphaNeuron returns the catalogue model-family defaults.
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

// Valid reports whether the complete numerical configuration is admissible.
func (s AlphaNeuronState) Valid() bool {
	return finite(s.V) &&
		finite(s.AExc) &&
		finite(s.IExc) &&
		finite(s.AInh) &&
		finite(s.IInh) &&
		finite(s.VRest) &&
		finite(s.VThreshold) &&
		s.VThreshold > s.VRest &&
		finite(s.TauV) && s.TauV > 0.0 &&
		finite(s.TauExc) && s.TauExc > 0.0 &&
		finite(s.TauInh) && s.TauInh > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0
}

// Step advances the exact constant-input flow. Rejected calls are atomic.
func (s *AlphaNeuronState) Step(excCurrent float64, inhCurrent float64) (int, error) {
	if !finite(excCurrent) || !finite(inhCurrent) || !s.Valid() {
		return 0, ErrAlphaInvalidState
	}
	aExcNext, iExcNext, err := alphaFilterCandidates(s.AExc, s.IExc, excCurrent, s.TauExc, s.Dt)
	if err != nil {
		return 0, err
	}
	aInhNext, iInhNext, err := alphaFilterCandidates(s.AInh, s.IInh, inhCurrent, s.TauInh, s.Dt)
	if err != nil {
		return 0, err
	}
	excSteady := s.TauExc * excCurrent
	inhSteady := s.TauInh * inhCurrent
	vSteady := s.VRest + excSteady - inhSteady
	decayV := math.Exp(-s.Dt / s.TauV)
	excContribution, err := alphaDriveContribution(s.IExc-excSteady, s.AExc-excSteady, s.TauExc, s.TauV, s.Dt)
	if err != nil {
		return 0, err
	}
	inhContribution, err := alphaDriveContribution(s.IInh-inhSteady, s.AInh-inhSteady, s.TauInh, s.TauV, s.Dt)
	if err != nil {
		return 0, err
	}
	vNext := vSteady + (s.V-vSteady)*decayV + excContribution - inhContribution
	if !finite(vNext) {
		return 0, ErrAlphaNonFiniteUpdate
	}
	s.AExc, s.IExc = aExcNext, iExcNext
	s.AInh, s.IInh = aInhNext, iInhNext
	if vNext >= s.VThreshold {
		s.V = s.VRest
		return 1, nil
	}
	s.V = vNext
	return 0, nil
}

// Reset restores the documented rest state without changing configuration.
func (s *AlphaNeuronState) Reset() {
	s.V = s.VRest
	s.AExc = 0.0
	s.IExc = 0.0
	s.AInh = 0.0
	s.IInh = 0.0
}

// SimulateAlphaNeuron returns the membrane trace and spike count.
func SimulateAlphaNeuron(nSteps int, excCurrent float64, inhCurrent float64) ([]float64, int) {
	if nSteps < 0 || !finite(excCurrent) || !finite(inhCurrent) {
		return nil, 0
	}
	s := NewAlphaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(excCurrent, inhCurrent)
		if err != nil {
			panic(err)
		}
		trace[t] = s.V
		spikes += result
	}
	return trace, spikes
}

var (
	ErrAlphaInvalidState    = errors.New("alpha state/current must be finite and well-formed")
	ErrAlphaNonFiniteUpdate = errors.New("alpha exact-flow update became non-finite")
)

func alphaFilterCandidates(
	riseState float64,
	currentState float64,
	drive float64,
	tau float64,
	dt float64,
) (float64, float64, error) {
	steadyState := tau * drive
	riseDelta := riseState - steadyState
	currentDelta := currentState - steadyState
	decay := math.Exp(-dt / tau)
	riseNext := steadyState + riseDelta*decay
	currentNext := steadyState + decay*(currentDelta+riseDelta*dt/tau)
	if !finite(riseNext) || !finite(currentNext) {
		return 0.0, 0.0, ErrAlphaNonFiniteUpdate
	}
	return riseNext, currentNext, nil
}

func alphaDriveContribution(
	currentDelta float64,
	riseDelta float64,
	tauDrive float64,
	tauV float64,
	dt float64,
) (float64, error) {
	rateV := 1.0 / tauV
	rateDrive := 1.0 / tauDrive
	decayV := math.Exp(-dt / tauV)
	decayDrive := math.Exp(-dt / tauDrive)
	var contribution float64
	if math.Abs(rateV-rateDrive) <= 1.0e-14 {
		contribution = rateV * decayV * (currentDelta*dt + riseDelta*dt*dt/(2.0*tauDrive))
	} else {
		rateDelta := rateV - rateDrive
		firstOrder := currentDelta * (decayDrive - decayV) / rateDelta
		secondOrder := riseDelta / tauDrive *
			(decayDrive*(rateDelta*dt-1.0) + decayV) /
			(rateDelta * rateDelta)
		contribution = rateV * (firstOrder + secondOrder)
	}
	if !finite(contribution) {
		return 0.0, ErrAlphaNonFiniteUpdate
	}
	return contribution, nil
}

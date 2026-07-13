// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for the published DPI neuron circuit

package services

import (
	"errors"
	"math"
)

// DPINeuronState holds the coupled current states and circuit parameters.
type DPINeuronState struct {
	IMem             float64
	IAHP             float64
	RefractoryTime   float64
	IThreshold       float64
	IReset           float64
	IRest            float64
	ITau             float64
	IG               float64
	ITauAHP          float64
	IGA              float64
	ISpike           float64
	I0               float64
	Kappa            float64
	Alpha            float64
	Tau              float64
	TauAHP           float64
	RefractoryPeriod float64
	Dt               float64
}

// NewDPINeuron creates a DPI neuron with the maintained normalised operating point.
func NewDPINeuron() *DPINeuronState {
	return &DPINeuronState{
		IMem:             0.01,
		IAHP:             0.01,
		RefractoryTime:   0.0,
		IThreshold:       1.0,
		IReset:           0.01,
		IRest:            0.1,
		ITau:             1.0,
		IG:               1.0,
		ITauAHP:          0.1,
		IGA:              1.0,
		ISpike:           5.0,
		I0:               0.01,
		Kappa:            0.7,
		Alpha:            10.0,
		Tau:              20.0,
		TauAHP:           100.0,
		RefractoryPeriod: 2.0,
		Dt:               0.1,
	}
}

func finiteDPI(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func positiveDPI(values ...float64) bool {
	for _, value := range values {
		if !finiteDPI(value) || value <= 0.0 {
			return false
		}
	}
	return true
}

// Valid reports whether the state lies inside the circuit's physical current domain.
func (s DPINeuronState) Valid() bool {
	return positiveDPI(
		s.IMem,
		s.IThreshold,
		s.IReset,
		s.ITau,
		s.IG,
		s.ITauAHP,
		s.IGA,
		s.ISpike,
		s.I0,
		s.Kappa,
		s.Alpha,
		s.Tau,
		s.TauAHP,
		s.RefractoryPeriod,
		s.Dt,
	) && finiteDPI(s.IAHP) && s.IAHP >= 0.0 &&
		finiteDPI(s.RefractoryTime) && s.RefractoryTime >= 0.0 &&
		finiteDPI(s.IRest) && s.IRest >= 0.0 &&
		s.IReset < s.IThreshold && s.RefractoryPeriod >= s.Dt
}

func sigmoidDPI(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	exponential := math.Exp(value)
	return exponential / (1.0 + exponential)
}

func (s DPINeuronState) feedbackCurrent() float64 {
	logCurrent := (math.Log(s.I0) + s.Kappa*math.Log(s.IMem)) / (s.Kappa + 1.0)
	gate := sigmoidDPI(s.Alpha * (s.IMem - s.IThreshold))
	return math.Exp(logCurrent) * gate
}

// Step advances one mutation-atomic Euler update of Indiveri et al. 2010 Eq. (3).
func (s *DPINeuronState) Step(current float64) (int, error) {
	totalInput := s.IRest + current
	if !finiteDPI(current) || !finiteDPI(totalInput) || !s.Valid() || totalInput < 0.0 {
		return 0, ErrDPIInvalidState
	}

	spikeActive := s.RefractoryTime > 0.0
	spikeCurrent := 0.0
	if spikeActive {
		spikeCurrent = s.ISpike
	}
	dIAHP := s.IAHP / (s.TauAHP * s.ITauAHP) *
		(spikeCurrent/(1.0+s.IAHP/s.IGA) - s.ITauAHP)
	nextIAHP := s.IAHP + s.Dt*dIAHP

	nextIMem := s.IReset
	nextRefractory := 0.0
	spiked := 0
	if spikeActive {
		nextRefractory = math.Max(0.0, s.RefractoryTime-s.Dt)
	} else {
		iFB := s.feedbackCurrent()
		dIMem := s.IMem / (s.Tau * s.ITau) *
			(totalInput/(1.0+s.IMem/s.IG) - s.ITau + iFB - s.IAHP)
		nextIMem = s.IMem + s.Dt*dIMem
		if !finiteDPI(nextIMem) || nextIMem <= 0.0 {
			return 0, ErrDPINonFiniteUpdate
		}
		if nextIMem >= s.IThreshold {
			nextIMem = s.IReset
			nextRefractory = s.RefractoryPeriod
			spiked = 1
		}
	}

	if !finiteDPI(nextIMem) || !finiteDPI(nextIAHP) || !finiteDPI(nextRefractory) ||
		nextIMem <= 0.0 || nextIAHP < 0.0 || nextRefractory < 0.0 {
		return 0, ErrDPINonFiniteUpdate
	}

	s.IMem = nextIMem
	s.IAHP = nextIAHP
	s.RefractoryTime = nextRefractory
	return spiked, nil
}

// Reset restores the leakage-current baseline without changing parameters.
func (s *DPINeuronState) Reset() {
	s.IMem = s.IReset
	s.IAHP = s.I0
	s.RefractoryTime = 0.0
}

// SimulateDPINeuron runs the maintained factory-default contract.
func SimulateDPINeuron(nSteps int, current float64) ([]float64, int) {
	trace, spikes, _, err := SimulateDPITrace(*NewDPINeuron(), nSteps, current)
	if err != nil {
		panic(err)
	}
	return trace, spikes
}

// SimulateDPITrace executes a complete circuit contract without partial output.
func SimulateDPITrace(
	initial DPINeuronState,
	nSteps int,
	current float64,
) ([]float64, int, DPINeuronState, error) {
	totalInput := initial.IRest + current
	if nSteps < 0 || !finiteDPI(current) || !finiteDPI(totalInput) ||
		!initial.Valid() || totalInput < 0.0 {
		return nil, 0, initial, ErrDPIInvalidState
	}
	state := initial
	trace := make([]float64, nSteps)
	spikes := 0
	for index := 0; index < nSteps; index++ {
		spike, err := state.Step(current)
		if err != nil {
			return nil, 0, initial, err
		}
		trace[index] = state.IMem
		spikes += spike
	}
	return trace, spikes, state, nil
}

var (
	ErrDPIInvalidState = errors.New(
		"DPI state/current must be finite and inside the physical current domain",
	)
	ErrDPINonFiniteUpdate = errors.New("DPI Euler update left the physical current domain")
)

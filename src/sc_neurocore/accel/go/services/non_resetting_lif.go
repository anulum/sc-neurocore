// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import (
	"errors"
	"math"
)

const (
	nonResettingLIFVMin     = -200.0
	nonResettingLIFVMax     = 200.0
	nonResettingLIFThetaMax = 1.0e9
)

// NonResettingLIFNeuronState contains the complete Kobayashi MAT(1) state and profile.
type NonResettingLIFNeuronState struct {
	V                   float64
	Theta               float64
	RefractoryRemaining float64
	Omega               float64
	TauM                float64
	TauTheta            float64
	Alpha               float64
	Resistance          float64
	RefractoryPeriod    float64
	Dt                  float64
}

// NewNonResettingLIFNeuron constructs the documented MAT(1) numerical specialization.
func NewNonResettingLIFNeuron() *NonResettingLIFNeuronState {
	return &NonResettingLIFNeuronState{V: 0.0, Theta: 0.0, RefractoryRemaining: 0.0, Omega: 19.0, TauM: 5.0, TauTheta: 50.0, Alpha: 37.0, Resistance: 50.0, RefractoryPeriod: 2.0, Dt: 0.001}
}

// Valid reports whether complete state and configuration satisfy the safety contract.
func (s NonResettingLIFNeuronState) Valid() bool {
	return nonResettingLIFFinite(s.V) && s.V >= nonResettingLIFVMin && s.V <= nonResettingLIFVMax &&
		nonResettingLIFFinite(s.Theta) && s.Theta >= 0.0 && s.Theta <= nonResettingLIFThetaMax &&
		nonResettingLIFFinite(s.RefractoryRemaining) && s.RefractoryRemaining >= 0.0 && s.RefractoryRemaining <= s.RefractoryPeriod &&
		nonResettingLIFFinite(s.Omega) && math.Abs(s.Omega) <= nonResettingLIFThetaMax &&
		nonResettingLIFFinite(s.TauM) && s.TauM > 0.0 && nonResettingLIFFinite(s.TauTheta) && s.TauTheta > 0.0 &&
		nonResettingLIFFinite(s.Alpha) && s.Alpha >= 0.0 && s.Alpha <= nonResettingLIFThetaMax &&
		nonResettingLIFFinite(s.Resistance) && s.Resistance > 0.0 && nonResettingLIFFinite(s.RefractoryPeriod) && s.RefractoryPeriod >= 0.0 &&
		nonResettingLIFFinite(s.Dt) && s.Dt > 0.0
}

// Step advances one atomic MAT(1) source-equation sample.
func (s *NonResettingLIFNeuronState) Step(current float64) (int, error) {
	if !nonResettingLIFFinite(current) || !s.Valid() {
		return 0, ErrNonResettingLIFInvalidState
	}
	nextV := s.V + s.Dt*(-s.V+s.Resistance*current)/s.TauM
	nextTheta := s.Theta * math.Exp(-s.Dt/s.TauTheta)
	nextRefractory := math.Max(0.0, s.RefractoryRemaining-s.Dt)
	if !nonResettingLIFFinite(nextV) || !nonResettingLIFFinite(nextTheta) || !nonResettingLIFFinite(nextRefractory) || nextV < nonResettingLIFVMin || nextV > nonResettingLIFVMax || nextTheta < 0.0 || nextTheta > nonResettingLIFThetaMax {
		return 0, ErrNonResettingLIFNonFiniteUpdate
	}
	spike := 0
	if nextRefractory == 0.0 && nextV >= s.Omega+nextTheta {
		spike = 1
		nextTheta += s.Alpha
		nextRefractory = s.RefractoryPeriod
	}
	if !nonResettingLIFFinite(nextTheta) || nextTheta > nonResettingLIFThetaMax {
		return 0, ErrNonResettingLIFNonFiniteUpdate
	}
	s.V, s.Theta, s.RefractoryRemaining = nextV, nextTheta, nextRefractory
	return spike, nil
}

// Reset restores zero-rest source state while retaining configuration.
func (s *NonResettingLIFNeuronState) Reset() { s.V, s.Theta, s.RefractoryRemaining = 0.0, 0.0, 0.0 }

// SimulateNonResettingLIFNeuron returns voltage and event count for a constant drive.
func SimulateNonResettingLIFNeuron(nSteps int, current float64) ([]float64, int) {
	s := NewNonResettingLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for i := range trace {
		event, err := s.Step(current)
		if err != nil {
			panic(err)
		}
		trace[i] = s.V
		spikes += event
	}
	return trace, spikes
}

var (
	// ErrNonResettingLIFInvalidState reports invalid source state, configuration, or input.
	ErrNonResettingLIFInvalidState = errors.New("non-resetting-lif MAT(1) state/current must be finite and physically valid")
	// ErrNonResettingLIFNonFiniteUpdate reports a candidate outside the safety envelope.
	ErrNonResettingLIFNonFiniteUpdate = errors.New("non-resetting-lif MAT(1) update left the safety envelope")
)

func nonResettingLIFFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

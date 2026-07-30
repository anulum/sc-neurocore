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

// SCNonResettingAdaptiveLIFNeuronState contains the retained SC recurrence.
type SCNonResettingAdaptiveLIFNeuronState struct{ V, Theta, VRest, ThetaRest, DeltaTheta, TauM, TauTheta, RM, Dt float64 }

// NewSCNonResettingAdaptiveLIFNeuron constructs the frozen project defaults.
func NewSCNonResettingAdaptiveLIFNeuron() *SCNonResettingAdaptiveLIFNeuronState {
	return &SCNonResettingAdaptiveLIFNeuronState{-65.0, -50.0, -65.0, -50.0, 5.0, 10.0, 50.0, 1.0, 0.1}
}

// Valid reports whether all project state and configuration are finite and valid.
func (s SCNonResettingAdaptiveLIFNeuronState) Valid() bool {
	values := []float64{s.V, s.Theta, s.VRest, s.ThetaRest, s.DeltaTheta, s.TauM, s.TauTheta, s.RM, s.Dt}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return s.DeltaTheta >= 0.0 && s.RM >= 0.0 && s.TauM > 0.0 && s.TauTheta > 0.0 && s.Dt > 0.0
}

// Step advances one atomic exact-relaxation project sample.
func (s *SCNonResettingAdaptiveLIFNeuronState) Step(current float64) (int, error) {
	if math.IsNaN(current) || math.IsInf(current, 0) || !s.Valid() {
		return 0, ErrSCNonResettingAdaptiveLIFInvalidState
	}
	steady := s.VRest + s.RM*current
	dv := math.Exp(-s.Dt / s.TauM)
	dt := math.Exp(-s.Dt / s.TauTheta)
	nextV := dv*s.V + (1.0-dv)*steady
	nextTheta := dt*s.Theta + (1.0-dt)*s.ThetaRest
	if math.IsNaN(steady) || math.IsInf(steady, 0) || math.IsNaN(nextV) || math.IsInf(nextV, 0) || math.IsNaN(nextTheta) || math.IsInf(nextTheta, 0) {
		return 0, ErrSCNonResettingAdaptiveLIFNonFiniteUpdate
	}
	spike := 0
	if nextV >= nextTheta {
		spike = 1
		nextTheta += s.DeltaTheta
	}
	if math.IsNaN(nextTheta) || math.IsInf(nextTheta, 0) {
		return 0, ErrSCNonResettingAdaptiveLIFNonFiniteUpdate
	}
	s.V, s.Theta = nextV, nextTheta
	return spike, nil
}

// Reset restores voltage and threshold to configured rests.
func (s *SCNonResettingAdaptiveLIFNeuronState) Reset() { s.V, s.Theta = s.VRest, s.ThetaRest }

var (
	// ErrSCNonResettingAdaptiveLIFInvalidState reports invalid project state, configuration, or input.
	ErrSCNonResettingAdaptiveLIFInvalidState = errors.New("SC non-resetting adaptive LIF state/current is invalid")
	// ErrSCNonResettingAdaptiveLIFNonFiniteUpdate reports a non-finite candidate.
	ErrSCNonResettingAdaptiveLIFNonFiniteUpdate = errors.New("SC non-resetting adaptive LIF update became non-finite")
)

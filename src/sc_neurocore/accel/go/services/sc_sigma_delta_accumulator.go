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

// SCSigmaDeltaAccumulatorState contains the frozen bipolar project state.
type SCSigmaDeltaAccumulatorState struct{ Sigma, VThreshold float64 }

// NewSCSigmaDeltaAccumulator constructs the project default.
func NewSCSigmaDeltaAccumulator() *SCSigmaDeltaAccumulatorState {
	return &SCSigmaDeltaAccumulatorState{0, 1}
}

// Valid reports whether project state is finite and configured.
func (s SCSigmaDeltaAccumulatorState) Valid() bool {
	return !math.IsNaN(s.Sigma) && !math.IsInf(s.Sigma, 0) && !math.IsNaN(s.VThreshold) && !math.IsInf(s.VThreshold, 0) && s.VThreshold > 0
}

// Step advances one atomic signed accumulator transition.
func (s *SCSigmaDeltaAccumulatorState) Step(current float64) (int, error) {
	if !s.Valid() || math.IsNaN(current) || math.IsInf(current, 0) {
		return 0, ErrSCSigmaDeltaInvalid
	}
	sigma := s.Sigma + current
	if math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		return 0, ErrSCSigmaDeltaInvalid
	}
	event := 0
	if sigma >= s.VThreshold {
		sigma -= s.VThreshold
		event = 1
	} else if sigma <= -s.VThreshold {
		sigma += s.VThreshold
		event = -1
	}
	s.Sigma = sigma
	return event, nil
}

// ErrSCSigmaDeltaInvalid reports an invalid project transition.
var ErrSCSigmaDeltaInvalid = errors.New("SC sigma-delta accumulator state or input is invalid")

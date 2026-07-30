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

const sigmaDeltaLimit = 1.0e12

// SigmaDeltaNeuronState contains the sampled APSDM prefilter and reconstruction state.
type SigmaDeltaNeuronState struct {
	Sigma, Reconstruction, Delta, TauReconstruction, Dt float64
}

// NewSigmaDeltaNeuron constructs the documented source specialization.
func NewSigmaDeltaNeuron() *SigmaDeltaNeuronState {
	return &SigmaDeltaNeuronState{0, 0, 1, 10, 0.1}
}

// Valid reports whether complete state and configuration are safe.
func (s SigmaDeltaNeuronState) Valid() bool {
	return finiteSD(s.Sigma) && finiteSD(s.Reconstruction) && finiteSD(s.Delta) && finiteSD(s.TauReconstruction) && finiteSD(s.Dt) && math.Abs(s.Sigma) <= sigmaDeltaLimit && math.Abs(s.Reconstruction) <= sigmaDeltaLimit && s.Delta > 0 && s.TauReconstruction > 0 && s.Dt > 0
}

// Step advances one atomic sampled APSDM transition.
func (s *SigmaDeltaNeuronState) Step(current float64) (int, error) {
	if !finiteSD(current) || !s.Valid() {
		return 0, ErrSigmaDeltaInvalid
	}
	sigma := s.Sigma + s.Dt*current
	reconstruction := s.Reconstruction * math.Exp(-s.Dt/s.TauReconstruction)
	spike := 0
	if sigma-reconstruction >= 0.5*s.Delta {
		spike = 1
		reconstruction += s.Delta
	}
	if !finiteSD(sigma) || !finiteSD(reconstruction) || math.Abs(sigma) > sigmaDeltaLimit || math.Abs(reconstruction) > sigmaDeltaLimit {
		return 0, ErrSigmaDeltaInvalid
	}
	s.Sigma, s.Reconstruction = sigma, reconstruction
	return spike, nil
}

// Reset clears both dynamic states.
func (s *SigmaDeltaNeuronState) Reset() { s.Sigma, s.Reconstruction = 0, 0 }

// ErrSigmaDeltaInvalid reports invalid state, input, or candidate.
var ErrSigmaDeltaInvalid = errors.New("sigma-delta state, configuration, input, or candidate is invalid")

func finiteSD(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

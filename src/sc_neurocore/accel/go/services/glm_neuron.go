// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for GLMNeuron

package services

import (
	"errors"
	"math"
	"math/rand"
)

// GLMNeuronState holds the complete point-process GLM state.
type GLMNeuronState struct {
	Mu       float64
	DtMs     float64
	K        []float64
	H        []float64
	StimBuf  []float64
	SpikeBuf []float64
}

// NewGLMNeuron creates a GLMNeuron with the canonical reference filters.
func NewGLMNeuron(nK, nH int) *GLMNeuronState {
	k := make([]float64, nK)
	for i := range k {
		k[i] = math.Exp(-float64(i)/3.0) * 0.5
	}
	h := make([]float64, nH)
	for t := range h {
		h[t] = -5.0*math.Exp(-float64(t)/2.0) + 0.5*math.Exp(-float64(t)/10.0)
	}
	return &GLMNeuronState{
		Mu:       -3.0,
		DtMs:     1.0,
		K:        k,
		H:        h,
		StimBuf:  make([]float64, nK),
		SpikeBuf: make([]float64, nH),
	}
}

func glmFinite(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// ValidGLMNeuron enforces the public runtime safety bounds.
func ValidGLMNeuron(s *GLMNeuronState) bool {
	if s == nil || math.IsNaN(s.Mu) || math.IsInf(s.Mu, 0) {
		return false
	}
	if math.IsNaN(s.DtMs) || math.IsInf(s.DtMs, 0) || s.DtMs <= 0.0 || s.DtMs > 1000.0 {
		return false
	}
	if len(s.K) != len(s.StimBuf) || len(s.H) != len(s.SpikeBuf) {
		return false
	}
	if len(s.K) == 0 || len(s.H) == 0 {
		return false
	}
	return glmFinite(s.K) && glmFinite(s.H) && glmFinite(s.StimBuf) && glmFinite(s.SpikeBuf)
}

// TryStep advances one step consuming an explicit uniform sample in [0, 1).
func (s *GLMNeuronState) TryStep(stimulus, uniform float64) (int, error) {
	if math.IsNaN(stimulus) || math.IsInf(stimulus, 0) {
		return 0, errors.New("stimulus must be finite")
	}
	if math.IsNaN(uniform) || math.IsInf(uniform, 0) || uniform < 0.0 || uniform >= 1.0 {
		return 0, errors.New("uniform must be finite and within [0, 1)")
	}
	if !ValidGLMNeuron(s) {
		return 0, errors.New("GLM state and parameters must satisfy the public bounds")
	}

	nK := len(s.StimBuf)
	nH := len(s.SpikeBuf)
	stimCandidate := make([]float64, nK)
	copy(stimCandidate, s.StimBuf)
	for i := nK - 1; i >= 1; i-- {
		stimCandidate[i] = stimCandidate[i-1]
	}
	stimCandidate[0] = stimulus

	dotK := 0.0
	for i := range s.K {
		dotK += s.K[i] * stimCandidate[i]
	}
	dotH := 0.0
	for i := range s.H {
		dotH += s.H[i] * s.SpikeBuf[i]
	}
	logRate := dotK + dotH + s.Mu
	if logRate > 20.0 {
		logRate = 20.0
	}
	if logRate < -20.0 {
		logRate = -20.0
	}
	p := math.Exp(logRate) * s.DtMs / 1000.0
	spike := 0
	if uniform < math.Min(p, 1.0) {
		spike = 1
	}
	spikeCandidate := make([]float64, nH)
	copy(spikeCandidate, s.SpikeBuf)
	for i := nH - 1; i >= 1; i-- {
		spikeCandidate[i] = spikeCandidate[i-1]
	}
	spikeCandidate[0] = float64(spike)

	s.StimBuf = stimCandidate
	s.SpikeBuf = spikeCandidate
	return spike, nil
}

// Reset restores the history buffers while preserving the filters.
func (s *GLMNeuronState) Reset() {
	for i := range s.StimBuf {
		s.StimBuf[i] = 0.0
	}
	for i := range s.SpikeBuf {
		s.SpikeBuf[i] = 0.0
	}
}

// SimulateGLMNeuron runs the neuron for n steps with a service-local
// seeded generator (regression evidence only, not a parity surface).
func SimulateGLMNeuron(nSteps int, stimulus float64, seed int64) ([]float64, int) {
	s := NewGLMNeuron(10, 20)
	rng := rand.New(rand.NewSource(seed)) // #nosec G404 -- simulation sampling, not security material.
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spike, err := s.TryStep(stimulus, rng.Float64())
		if err != nil {
			return trace, spikes
		}
		spikes += spike
		trace[t] = s.SpikeBuf[0]
	}
	return trace, spikes
}

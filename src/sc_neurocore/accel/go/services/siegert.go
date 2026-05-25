// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for siegert

package services

import (
	"errors"
	"math"
)

// SiegertTransferFunctionState holds the neuron state
type SiegertTransferFunctionState struct {
	TauM       float64
	TauRp      float64
	VThreshold float64
	VReset     float64
	VRest      float64
}

// NewSiegertTransferFunction creates a new SiegertTransferFunction neuron with default parameters
func NewSiegertTransferFunction() *SiegertTransferFunctionState {
	return &SiegertTransferFunctionState{
		TauM:       20.0,
		TauRp:      2.0,
		VThreshold: -50.0,
		VReset:     -70.0,
		VRest:      -65.0,
	}
}

// Step evaluates the refractory-bounded Siegert transfer rate in Hz.
func (s *SiegertTransferFunctionState) Step(iExt float64) (float64, error) {
	if !ValidateSiegert(s) || !siegertFinite(iExt) {
		return 0, ErrSiegertInvalidState
	}
	mu := s.VRest + iExt
	if !siegertFinite(mu) {
		return 0, ErrSiegertNonFiniteOutput
	}
	sigma := math.Max(math.Abs(iExt)*0.1, 1.0e-6)
	if !siegertFinite(sigma) || sigma <= 0.0 {
		return 0, ErrSiegertNonFiniteOutput
	}
	uTh := (s.VThreshold - mu) / sigma
	uRe := (s.VReset - mu) / sigma
	if !siegertFinite(uTh) || !siegertFinite(uRe) || uTh <= uRe {
		return 0, ErrSiegertNonFiniteOutput
	}
	half := 0.5 * (uTh - uRe)
	mid := 0.5 * (uTh + uRe)
	if !siegertFinite(half) || !siegertFinite(mid) || half <= 0.0 {
		return 0, ErrSiegertNonFiniteOutput
	}
	integral := 0.0
	for i, node := range siegertNodes20 {
		u := half*node + mid
		erfU := siegertErfApprox(u)
		integrand := math.Exp(math.Min(u*u, 50.0)) * (1.0 + erfU)
		if !siegertFinite(integrand) {
			return 0, ErrSiegertNonFiniteOutput
		}
		integral += siegertWeights20[i] * integrand
	}
	integral *= half
	if !siegertFinite(integral) || integral < 0.0 {
		return 0, ErrSiegertNonFiniteOutput
	}
	tISI := s.TauRp + s.TauM*math.Sqrt(math.Pi)*integral
	if !siegertFinite(tISI) || tISI < s.TauRp {
		return 0, ErrSiegertNonFiniteOutput
	}
	rate := 1000.0 / tISI
	maxRate := 1000.0 / s.TauRp
	if !siegertFinite(rate) || rate < 0.0 || rate > maxRate {
		return 0, ErrSiegertNonFiniteOutput
	}
	return rate, nil
}

// ValidateSiegert enforces the first-passage boundary contract.
func ValidateSiegert(s *SiegertTransferFunctionState) bool {
	if s == nil {
		return false
	}
	return siegertFinite(s.TauM) && s.TauM > 0.0 && siegertFinite(s.TauRp) && s.TauRp > 0.0 &&
		siegertFinite(s.VThreshold) && siegertFinite(s.VReset) && siegertFinite(s.VRest) && s.VThreshold > s.VReset
}

// SimulateSiegertTransferFunction runs the neuron for n steps
func SimulateSiegertTransferFunction(nSteps int, iExt float64) ([]float64, int) {
	s := NewSiegertTransferFunction()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = result
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrSiegertInvalidState    = errors.New("siegert state/current must be finite and physically ordered")
	ErrSiegertNonFiniteOutput = errors.New("siegert first-passage calculation became non-finite or unbounded")
)

var siegertNodes20 = [20]float64{-0.993128599185095, -0.963971927277914, -0.912234428251326, -0.839116971822219, -0.746331906460151, -0.636053680726515, -0.510867001950827, -0.37370608871542, -0.227785851141645, -0.076526521133497, 0.076526521133497, 0.227785851141645, 0.37370608871542, 0.510867001950827, 0.636053680726515, 0.746331906460151, 0.839116971822219, 0.912234428251326, 0.963971927277914, 0.993128599185095}
var siegertWeights20 = [20]float64{0.017614007139152, 0.040601429800387, 0.062672048334109, 0.083276741576704, 0.10193011981724, 0.118194531961518, 0.131688638449177, 0.142096109318382, 0.149172986472604, 0.152753387130726, 0.152753387130726, 0.149172986472604, 0.142096109318382, 0.131688638449177, 0.118194531961518, 0.10193011981724, 0.083276741576704, 0.062672048334109, 0.040601429800387, 0.017614007139152}

func siegertFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func siegertErfApprox(x float64) float64 {
	sign := 1.0
	if x < 0.0 {
		sign = -1.0
	}
	a := math.Abs(x)
	t := 1.0 / (1.0 + 0.3275911*a)
	poly := t * (0.254829592 + t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))
	return sign * (1.0 - poly*math.Exp(-a*a))
}

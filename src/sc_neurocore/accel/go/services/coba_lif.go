// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for coba_lif

package services

import (
	"errors"
	"math"
)

const cobaLIFVMin = -200.0
const cobaLIFVMax = 100.0
const cobaLIFGMax = 1.0e9

// COBALIFNeuronState holds the neuron state.
type COBALIFNeuronState struct {
	V          float64
	GE         float64
	GI         float64
	CM         float64
	GL         float64
	EL         float64
	EE         float64
	EI         float64
	TauE       float64
	TauI       float64
	VThreshold float64
	VReset     float64
	Dt         float64
}

// NewCOBALIFNeuron creates a new COBALIFNeuron neuron with default parameters.
func NewCOBALIFNeuron() *COBALIFNeuronState {
	return &COBALIFNeuronState{V: -65.0, GE: 0.0, GI: 0.0, CM: 200.0, GL: 10.0, EL: -65.0, EE: 0.0, EI: -80.0, TauE: 5.0, TauI: 10.0, VThreshold: -50.0, VReset: -65.0, Dt: 0.1}
}

func cobaLIFFinite(value float64) bool      { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func cobaLIFNonnegative(value float64) bool { return cobaLIFFinite(value) && value >= 0.0 }

func (s *COBALIFNeuronState) validate() error {
	if !cobaLIFFinite(s.V) || s.V < cobaLIFVMin || s.V > cobaLIFVMax {
		return errors.New("v outside COBA LIF safety envelope")
	}
	if !cobaLIFNonnegative(s.GE) || !cobaLIFNonnegative(s.GI) || s.GE > cobaLIFGMax || s.GI > cobaLIFGMax {
		return errors.New("conductance outside COBA LIF safety envelope")
	}
	for _, value := range []float64{s.CM, s.TauE, s.TauI, s.Dt} {
		if !cobaLIFFinite(value) || value <= 0.0 {
			return errors.New("positive COBA LIF parameter invalid")
		}
	}
	if !cobaLIFNonnegative(s.GL) {
		return errors.New("leak conductance invalid")
	}
	for _, value := range []float64{s.EL, s.EE, s.EI, s.VThreshold, s.VReset} {
		if !cobaLIFFinite(value) {
			return errors.New("finite COBA LIF parameter invalid")
		}
	}
	if s.VReset < cobaLIFVMin || s.VReset > cobaLIFVMax {
		return errors.New("v_reset outside COBA LIF safety envelope")
	}
	return nil
}

func (s *COBALIFNeuronState) derivatives(v, ge, gi, iExt float64) (float64, float64, float64) {
	iSyn := ge*(v-s.EE) + gi*(v-s.EI)
	dv := (-s.GL*(v-s.EL) - iSyn + iExt) / s.CM
	return dv, -ge / s.TauE, -gi / s.TauI
}

func (s *COBALIFNeuronState) rk4Candidate(v, ge, gi, iExt float64) (float64, float64, float64) {
	k1v, k1e, k1i := s.derivatives(v, ge, gi, iExt)
	k2v, k2e, k2i := s.derivatives(v+0.5*s.Dt*k1v, ge+0.5*s.Dt*k1e, gi+0.5*s.Dt*k1i, iExt)
	k3v, k3e, k3i := s.derivatives(v+0.5*s.Dt*k2v, ge+0.5*s.Dt*k2e, gi+0.5*s.Dt*k2i, iExt)
	k4v, k4e, k4i := s.derivatives(v+s.Dt*k3v, ge+s.Dt*k3e, gi+s.Dt*k3i, iExt)
	return v + (s.Dt/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v),
		ge + (s.Dt/6.0)*(k1e+2.0*k2e+2.0*k3e+k4e),
		gi + (s.Dt/6.0)*(k1i+2.0*k2i+2.0*k3i+k4i)
}

// Step advances the neuron by one timestep with current-only drive.
func (s *COBALIFNeuronState) Step(iExt float64) (int, error) {
	return s.StepWithConductance(iExt, 0.0, 0.0)
}

// StepWithConductance advances the neuron with excitatory and inhibitory conductance injections.
func (s *COBALIFNeuronState) StepWithConductance(iExt, deltaGE, deltaGI float64) (int, error) {
	if !cobaLIFFinite(iExt) || !cobaLIFNonnegative(deltaGE) || !cobaLIFNonnegative(deltaGI) {
		return 0, errors.New("invalid COBA LIF step input")
	}
	if err := s.validate(); err != nil {
		return 0, err
	}
	gePre := s.GE + deltaGE
	giPre := s.GI + deltaGI
	if gePre > cobaLIFGMax || giPre > cobaLIFGMax {
		return 0, errors.New("conductance candidate outside COBA LIF safety envelope")
	}
	vCandidate, geCandidate, giCandidate := s.rk4Candidate(s.V, gePre, giPre, iExt)
	iSyn := gePre*(s.V-s.EE) + giPre*(s.V-s.EI)
	if !cobaLIFFinite(iSyn) || !cobaLIFFinite(vCandidate) || !cobaLIFFinite(geCandidate) || !cobaLIFFinite(giCandidate) {
		return 0, errors.New("COBA LIF candidate must be finite")
	}
	if vCandidate < cobaLIFVMin || vCandidate > cobaLIFVMax {
		return 0, errors.New("voltage candidate outside COBA LIF safety envelope")
	}
	if geCandidate < 0.0 || giCandidate < 0.0 {
		return 0, errors.New("conductance candidate must remain non-negative")
	}
	if vCandidate >= s.VThreshold {
		s.V = s.VReset
		s.GE = geCandidate
		s.GI = giCandidate
		return 1, nil
	}
	s.V = vCandidate
	s.GE = geCandidate
	s.GI = giCandidate
	return 0, nil
}

// SimulateCOBALIFNeuron runs the neuron for n steps.
func SimulateCOBALIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewCOBALIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

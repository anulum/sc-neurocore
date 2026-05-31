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

func cobaLIFDecay(dt, tau float64) (float64, error) {
	ratio := -dt / tau
	if ratio < -700.0 {
		return 0.0, nil
	}
	decay := math.Exp(ratio)
	if !cobaLIFFinite(decay) || decay < 0.0 || decay >= 1.0 {
		return 0, errors.New("decay must be in [0, 1)")
	}
	return decay, nil
}

func (s *COBALIFNeuronState) validate() (float64, float64, error) {
	if !cobaLIFFinite(s.V) || s.V < cobaLIFVMin || s.V > cobaLIFVMax {
		return 0, 0, errors.New("v outside COBA LIF safety envelope")
	}
	if !cobaLIFNonnegative(s.GE) || !cobaLIFNonnegative(s.GI) || s.GE > cobaLIFGMax || s.GI > cobaLIFGMax {
		return 0, 0, errors.New("conductance outside COBA LIF safety envelope")
	}
	for _, value := range []float64{s.CM, s.TauE, s.TauI, s.Dt} {
		if !cobaLIFFinite(value) || value <= 0.0 {
			return 0, 0, errors.New("positive COBA LIF parameter invalid")
		}
	}
	if !cobaLIFNonnegative(s.GL) {
		return 0, 0, errors.New("leak conductance invalid")
	}
	for _, value := range []float64{s.EL, s.EE, s.EI, s.VThreshold, s.VReset} {
		if !cobaLIFFinite(value) {
			return 0, 0, errors.New("finite COBA LIF parameter invalid")
		}
	}
	if s.VReset < cobaLIFVMin || s.VReset > cobaLIFVMax {
		return 0, 0, errors.New("v_reset outside COBA LIF safety envelope")
	}
	decayE, err := cobaLIFDecay(s.Dt, s.TauE)
	if err != nil {
		return 0, 0, err
	}
	decayI, err := cobaLIFDecay(s.Dt, s.TauI)
	if err != nil {
		return 0, 0, err
	}
	return decayE, decayI, nil
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
	decayE, decayI, err := s.validate()
	if err != nil {
		return 0, err
	}
	gePre := s.GE + deltaGE
	giPre := s.GI + deltaGI
	if gePre > cobaLIFGMax || giPre > cobaLIFGMax {
		return 0, errors.New("conductance candidate outside COBA LIF safety envelope")
	}
	iSyn := gePre*(s.V-s.EE) + giPre*(s.V-s.EI)
	dv := (-s.GL*(s.V-s.EL) - iSyn + iExt) / s.CM * s.Dt
	vCandidate := s.V + dv
	geCandidate := gePre * decayE
	giCandidate := giPre * decayI
	if !cobaLIFFinite(iSyn) || !cobaLIFFinite(dv) || !cobaLIFFinite(vCandidate) || !cobaLIFFinite(geCandidate) || !cobaLIFFinite(giCandidate) {
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

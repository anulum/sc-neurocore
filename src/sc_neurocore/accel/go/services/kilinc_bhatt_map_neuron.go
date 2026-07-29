// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — deprecated Go compatibility identity

package services

// KilincBhattMapNeuronState is a compatibility alias for the retained SC map.
type KilincBhattMapNeuronState = SCAdaptiveThresholdMapNeuronState

// NewKilincBhattMapNeuron preserves the former constructor without false provenance.
func NewKilincBhattMapNeuron() *KilincBhattMapNeuronState {
	return NewSCAdaptiveThresholdMapNeuron()
}

// SimulateKilincBhattMapNeuron preserves the legacy constant-drive helper.
func SimulateKilincBhattMapNeuron(nSteps int, current float64) ([]float64, int) {
	state := NewSCAdaptiveThresholdMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for index := range trace {
		event := state.Step(current)
		trace[index] = state.X
		spikes += event
	}
	return trace, spikes
}

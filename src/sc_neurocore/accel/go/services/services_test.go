// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests

package services

import "testing"

func TestHodgkinHuxleySimulate(t *testing.T) {
	trace, spikes := SimulateHodgkinHuxleyNeuron(100, 10.0)
	if len(trace) != 100 {
		t.Errorf("expected 100 steps, got %d", len(trace))
	}
	_ = spikes
}

func TestAdExSimulate(t *testing.T) {
	trace, spikes := SimulateAdExNeuron(100, 10.0)
	if len(trace) != 100 {
		t.Errorf("expected 100 steps, got %d", len(trace))
	}
	_ = spikes
}

func TestFitzHughNagumoSimulate(t *testing.T) {
	trace, spikes := SimulateFitzHughNagumoNeuron(100, 10.0)
	if len(trace) != 100 {
		t.Errorf("expected 100 steps, got %d", len(trace))
	}
	_ = spikes
}

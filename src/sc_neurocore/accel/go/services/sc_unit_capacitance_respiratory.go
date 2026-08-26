// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained unit-capacitance respiratory recurrence

package services

// NewSCUnitCapacitanceRespiratoryNeuron returns the historical SC profile.
func NewSCUnitCapacitanceRespiratoryNeuron() *ButeraRespiratoryNeuronState {
	state := NewButeraRespiratoryNeuron()
	state.Capacitance = 1.0
	state.ESyn = -10.0
	return state
}

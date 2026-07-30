// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
package main

/*
#include <stdint.h>
*/
import "C"
import (
	"github.com/anulum/sc-neurocore/accel/services"
	"unsafe"
)

// energy_lif_simulate_c executes the complete pinned Fardet-Levina batch.
//
//export energy_lif_simulate_c
func energy_lif_simulate_c(stepsC C.int, vC, eC, cC, gC, e0C, euC, edC, efC, vthC, vresetC, alphaC, epsilon0C, epsilonCC, deltaC, tauEC, dtC C.double, currentsPtr, voltagesPtr, energiesPtr, eventsPtr, vfinalPtr, efinalPtr unsafe.Pointer) C.int {
	steps := int(stepsC)
	if steps < 0 || currentsPtr == nil || voltagesPtr == nil || energiesPtr == nil || eventsPtr == nil || vfinalPtr == nil || efinalPtr == nil {
		return 1
	}
	s := services.EnergyLIFNeuronState{V: float64(vC), Epsilon: float64(eC), Capacitance: float64(cC), GLeak: float64(gC), E0: float64(e0C), EU: float64(euC), ED: float64(edC), EF: float64(efC), VThreshold: float64(vthC), VReset: float64(vresetC), Alpha: float64(alphaC), Epsilon0: float64(epsilon0C), EpsilonC: float64(epsilonCC), Delta: float64(deltaC), TauE: float64(tauEC), Dt: float64(dtC)}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	voltages := unsafe.Slice((*C.double)(voltagesPtr), steps)
	energies := unsafe.Slice((*C.double)(energiesPtr), steps)
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for i := 0; i < steps; i++ {
		event := s.Step(float64(currents[i]))
		if event < 0 {
			return 2
		}
		voltages[i] = C.double(s.V)
		energies[i] = C.double(s.Epsilon)
		events[i] = C.int64_t(event)
	}
	*(*C.double)(vfinalPtr) = C.double(s.V)
	*(*C.double)(efinalPtr) = C.double(s.Epsilon)
	return 0
}
func main() {}

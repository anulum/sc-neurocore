// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go C ABI for complete Compte batches

// Package main exports the source-bounded Compte batch contract.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/anulum/sc-neurocore/accel/services"
)

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// compte_wm_simulate_c returns zero only after writing a complete valid batch.
// The three event arrays are recurrent NMDA, external AMPA, and inhibitory
// GABAA inputs. Output state order is membrane, AMPA, NMDA, NMDA precursor,
// GABAA, and refractory duration.
//
// Build: go build -buildmode=c-shared -o libcompte_wm.so compte_wm.go
//
//export compte_wm_simulate_c
func compte_wm_simulate_c(
	stepsC C.int,
	vC, sAmpaC, sNmdaC, xNmdaC, sGabaC, refC C.double,
	gLC, gAmpaC, gNmdaC, gGabaC, eLC, eExcC, eInhC, cMC, mgC C.double,
	tauAmpaC, tauNmdaC, tauXC, tauGabaC, alphaNmdaC C.double,
	vThresholdC, vResetC, tauRefC, dtC C.double,
	currentsPtr, recurrentPtr, externalPtr, inhibitoryPtr unsafe.Pointer,
	voltagesPtr, sAmpaOutPtr, sNmdaOutPtr, xNmdaOutPtr, sGabaOutPtr unsafe.Pointer,
	refractoryPtr, eventsPtr unsafe.Pointer,
	vFinalPtr, sAmpaFinalPtr, sNmdaFinalPtr, xNmdaFinalPtr, sGabaFinalPtr unsafe.Pointer,
	refFinalPtr unsafe.Pointer,
) C.int {
	steps := int(stepsC)
	if steps < 0 {
		return 1
	}
	values := []float64{
		float64(vC), float64(sAmpaC), float64(sNmdaC), float64(xNmdaC),
		float64(sGabaC), float64(refC), float64(gLC), float64(gAmpaC),
		float64(gNmdaC), float64(gGabaC), float64(eLC), float64(eExcC),
		float64(eInhC), float64(cMC), float64(mgC), float64(tauAmpaC),
		float64(tauNmdaC), float64(tauXC), float64(tauGabaC),
		float64(alphaNmdaC), float64(vThresholdC), float64(vResetC),
		float64(tauRefC), float64(dtC),
	}
	state := &services.CompteWMNeuronState{
		V: values[0], SAmpa: values[1], SNmda: values[2], XNmda: values[3],
		SGaba: values[4], RefRemaining: values[5], GL: values[6],
		GAmpa: values[7], GNmda: values[8], GGaba: values[9], EL: values[10],
		EExc: values[11], EInh: values[12], CM: values[13], Mg: values[14],
		TauAmpa: values[15], TauNmda: values[16], TauX: values[17],
		TauGaba: values[18], AlphaNmda: values[19], VThreshold: values[20],
		VReset: values[21], TauRef: values[22], Dt: values[23],
	}
	if !services.ValidateCompteWM(state) {
		return 2
	}
	currents := unsafe.Slice((*C.double)(currentsPtr), steps)
	recurrent := unsafe.Slice((*C.int64_t)(recurrentPtr), steps)
	external := unsafe.Slice((*C.int64_t)(externalPtr), steps)
	inhibitory := unsafe.Slice((*C.int64_t)(inhibitoryPtr), steps)
	for index := 0; index < steps; index++ {
		if !finite(float64(currents[index])) ||
			(recurrent[index] != 0 && recurrent[index] != 1) ||
			(external[index] != 0 && external[index] != 1) ||
			(inhibitory[index] != 0 && inhibitory[index] != 1) {
			return 3
		}
	}
	outputs := [][]C.double{
		unsafe.Slice((*C.double)(voltagesPtr), steps),
		unsafe.Slice((*C.double)(sAmpaOutPtr), steps),
		unsafe.Slice((*C.double)(sNmdaOutPtr), steps),
		unsafe.Slice((*C.double)(xNmdaOutPtr), steps),
		unsafe.Slice((*C.double)(sGabaOutPtr), steps),
		unsafe.Slice((*C.double)(refractoryPtr), steps),
	}
	events := unsafe.Slice((*C.int64_t)(eventsPtr), steps)
	for index := 0; index < steps; index++ {
		event, err := state.StepWithEvents(
			float64(currents[index]), recurrent[index] == 1,
			external[index] == 1, inhibitory[index] == 1,
		)
		if err != nil {
			return 4
		}
		dynamic := state.GetState()
		for output := range outputs {
			outputs[output][index] = C.double(dynamic[output])
		}
		events[index] = C.int64_t(event)
	}
	final := state.GetState()
	finalPointers := []unsafe.Pointer{
		vFinalPtr, sAmpaFinalPtr, sNmdaFinalPtr, xNmdaFinalPtr, sGabaFinalPtr, refFinalPtr,
	}
	for index, pointer := range finalPointers {
		*(*C.double)(pointer) = C.double(final[index])
	}
	return 0
}

func main() {}

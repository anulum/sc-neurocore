// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go cgo acceleration for network/gamma_oscillation
//
// Accelerator kernel for the PING gamma oscillation circuit.
// Implements the step logic matching the conductance-based dynamics 
// described in:
// Börgers, C. & Kopell, N. (2003). Synchronization in Networks of 
// Excitatory and Inhibitory Neurons with Sparse, Random Connectivity. 
// Neural Computation 15(3): 509-538.

package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

//export py_ping_step_c
func py_ping_step_c(
	n_excit C.int,
	n_inhib C.int,
	v_e *C.double,
	g_ampa_e *C.double,
	g_gaba_e *C.double,
	refrac_e *C.double,
	i_drive_e *C.double,
	xi_e *C.double,
	spikes_e_out *C.uint8_t,
	v_i *C.double,
	g_ampa_i *C.double,
	g_gaba_i *C.double,
	refrac_i *C.double,
	i_drive_i *C.double,
	xi_i *C.double,
	spikes_i_out *C.uint8_t,
	e_l C.double,
	e_ampa C.double,
	e_gaba C.double,
	g_l C.double,
	c_m C.double,
	v_threshold C.double,
	v_reset C.double,
	t_refrac C.double,
	tau_ampa C.double,
	tau_gaba C.double,
	sigma_e C.double,
	sigma_i C.double,
	dt C.double,
	out_n_e_spikes *C.uint32_t,
	out_n_i_spikes *C.uint32_t,
) {
	decay_ampa := math.Exp(-float64(dt) / float64(tau_ampa))
	decay_gaba := math.Exp(-float64(dt) / float64(tau_gaba))
	dt_over_cm := float64(dt) / float64(c_m)
	sqrt_dt := math.Sqrt(float64(dt))

	v_e_slice := unsafe.Slice((*float64)(v_e), int(n_excit))
	g_ampa_e_slice := unsafe.Slice((*float64)(g_ampa_e), int(n_excit))
	g_gaba_e_slice := unsafe.Slice((*float64)(g_gaba_e), int(n_excit))
	refrac_e_slice := unsafe.Slice((*float64)(refrac_e), int(n_excit))
	i_drive_e_slice := unsafe.Slice((*float64)(i_drive_e), int(n_excit))
	xi_e_slice := unsafe.Slice((*float64)(xi_e), int(n_excit))
	spikes_e_out_slice := unsafe.Slice((*uint8)(unsafe.Pointer(spikes_e_out)), int(n_excit))

	v_i_slice := unsafe.Slice((*float64)(v_i), int(n_inhib))
	g_ampa_i_slice := unsafe.Slice((*float64)(g_ampa_i), int(n_inhib))
	g_gaba_i_slice := unsafe.Slice((*float64)(g_gaba_i), int(n_inhib))
	refrac_i_slice := unsafe.Slice((*float64)(refrac_i), int(n_inhib))
	i_drive_i_slice := unsafe.Slice((*float64)(i_drive_i), int(n_inhib))
	xi_i_slice := unsafe.Slice((*float64)(xi_i), int(n_inhib))
	spikes_i_out_slice := unsafe.Slice((*uint8)(unsafe.Pointer(spikes_i_out)), int(n_inhib))

	for k := 0; k < int(n_excit); k++ {
		g_ampa_e_slice[k] *= decay_ampa
		g_gaba_e_slice[k] *= decay_gaba
	}
	for k := 0; k < int(n_inhib); k++ {
		g_ampa_i_slice[k] *= decay_ampa
		g_gaba_i_slice[k] *= decay_gaba
	}

	n_e := uint32(0)
	for k := 0; k < int(n_excit); k++ {
		in_refrac := refrac_e_slice[k] > 0.0
		v_old := v_e_slice[k]
		i_leak := -float64(g_l) * (v_old - float64(e_l))
		i_ampa_cur := -g_ampa_e_slice[k] * (v_old - float64(e_ampa))
		i_gaba_cur := -g_gaba_e_slice[k] * (v_old - float64(e_gaba))
		i_total := i_leak + i_ampa_cur + i_gaba_cur + i_drive_e_slice[k]
		noise := sqrt_dt * float64(sigma_e) * xi_e_slice[k]

		var v_new float64
		if in_refrac {
			v_new = float64(v_reset)
		} else {
			v_new = v_old + i_total*dt_over_cm + noise
		}

		spk := (v_new >= float64(v_threshold)) && !in_refrac

		if spk {
			v_e_slice[k] = float64(v_reset)
			// Apply new refractory. Also decay if not spiking handled below
			refrac_e_slice[k] = float64(t_refrac)
			spikes_e_out_slice[k] = 1
			n_e++
		} else {
			v_e_slice[k] = v_new
			new_refrac := refrac_e_slice[k] - float64(dt)
			if new_refrac > 0.0 {
				refrac_e_slice[k] = new_refrac
			} else {
				refrac_e_slice[k] = 0.0
			}
			spikes_e_out_slice[k] = 0
		}
	}
	*out_n_e_spikes = C.uint32_t(n_e)

	n_i := uint32(0)
	for k := 0; k < int(n_inhib); k++ {
		in_refrac := refrac_i_slice[k] > 0.0
		v_old := v_i_slice[k]
		i_leak := -float64(g_l) * (v_old - float64(e_l))
		i_ampa_cur := -g_ampa_i_slice[k] * (v_old - float64(e_ampa))
		i_gaba_cur := -g_gaba_i_slice[k] * (v_old - float64(e_gaba))
		i_total := i_leak + i_ampa_cur + i_gaba_cur + i_drive_i_slice[k]
		noise := sqrt_dt * float64(sigma_i) * xi_i_slice[k]

		var v_new float64
		if in_refrac {
			v_new = float64(v_reset)
		} else {
			v_new = v_old + i_total*dt_over_cm + noise
		}

		spk := (v_new >= float64(v_threshold)) && !in_refrac

		if spk {
			v_i_slice[k] = float64(v_reset)
			refrac_i_slice[k] = float64(t_refrac)
			spikes_i_out_slice[k] = 1
			n_i++
		} else {
			v_i_slice[k] = v_new
			new_refrac := refrac_i_slice[k] - float64(dt)
			if new_refrac > 0.0 {
				refrac_i_slice[k] = new_refrac
			} else {
				refrac_i_slice[k] = 0.0
			}
			spikes_i_out_slice[k] = 0
		}
	}
	*out_n_i_spikes = C.uint32_t(n_i)
}

func main() {}

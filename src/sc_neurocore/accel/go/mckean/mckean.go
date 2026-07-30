// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
package main

/*
#include <stdint.h>
typedef struct { double v; double w; int32_t event; int32_t status; } mckean_out;
*/
import "C"
import "github.com/anulum/sc-neurocore/accel/services"

//export mckean_step
func mckean_step(v, w, a, lambda, mu, b, dt, current C.double) C.mckean_out {
	s := services.McKeanNeuronState{V: float64(v), W: float64(w), A: float64(a), Lambda: float64(lambda), Mu: float64(mu), B: float64(b), Dt: float64(dt)}
	event := s.Step(float64(current))
	status := int32(0)
	if event < 0 {
		status = -1
	}
	return C.mckean_out{v: C.double(s.V), w: C.double(s.W), event: C.int32_t(event), status: C.int32_t(status)}
}
func main() {}

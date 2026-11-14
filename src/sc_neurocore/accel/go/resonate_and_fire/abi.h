// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stable resonate-and-fire Go C ABI

#ifndef SC_NEUROCORE_RESONATE_AND_FIRE_H
#define SC_NEUROCORE_RESONATE_AND_FIRE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int32_t resonate_and_fire_simulate_c(
    int32_t n,
    double x_init,
    double y_init,
    double b,
    double omega,
    double threshold,
    double dt,
    void *current,
    void *x_out,
    void *y_out,
    void *spikes_out,
    double *x_final,
    double *y_final,
    double *spike_count
);

#ifdef __cplusplus
}
#endif

#endif

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stable adaptive-threshold Go C ABI

#ifndef SC_NEUROCORE_ADAPTIVE_THRESHOLD_IF_H
#define SC_NEUROCORE_ADAPTIVE_THRESHOLD_IF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int32_t adaptive_threshold_if_simulate_c(
    int32_t n,
    double v_init,
    double theta_init,
    double v_rest,
    double v_reset,
    double theta_rest,
    double delta_theta,
    double tau_m,
    double tau_theta,
    double dt,
    void *current,
    void *v_out,
    void *theta_out,
    void *spikes_out,
    double *v_final,
    double *theta_final,
    double *spike_count
);

#ifdef __cplusplus
}
#endif

#endif

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stable alpha-synapse Go C ABI

#ifndef SC_NEUROCORE_ALPHA_H
#define SC_NEUROCORE_ALPHA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int32_t alpha_simulate_c(
    int32_t n,
    double v_init,
    double a_exc_init,
    double i_exc_init,
    double a_inh_init,
    double i_inh_init,
    double v_rest,
    double v_threshold,
    double tau_v,
    double tau_exc,
    double tau_inh,
    double dt,
    void *exc_current,
    void *inh_current,
    void *v_out,
    void *a_exc_out,
    void *i_exc_out,
    void *a_inh_out,
    void *i_inh_out,
    void *spikes_out,
    double *v_final,
    double *a_exc_final,
    double *i_exc_final,
    double *a_inh_final,
    double *i_inh_final,
    double *spike_count
);

#ifdef __cplusplus
}
#endif

#endif

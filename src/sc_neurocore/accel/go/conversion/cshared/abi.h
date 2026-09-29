// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense IF ABI-one C descriptors

#ifndef SC_IF_GO_ABI_H
#define SC_IF_GO_ABI_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* A layer borrows all arrays during replay; positive lengths require complete,
 * live, aligned arrays unchanged during the call. Address checks cannot prove
 * OS accessibility. Row-major storage matches dense-if-f64-sequential-v1. */
typedef struct {
    size_t outputs; /* Positive output width. */
    size_t inputs; /* Positive input width. */
    double *weights; /* outputs*inputs finite row-major doubles. */
    double *bias; /* Optional finite per-output biases; absent at zero length. */
    size_t bias_len; /* Zero or outputs. */
    double threshold; /* Positive finite inclusive threshold. */
    double initial_fraction; /* Finite membrane preload in threshold units. */
    double *initial; /* Optional batch*outputs state with flag8. */
    size_t initial_len; /* State length, ignored without flag8. */
} sc_if_layer;

/* Each request borrows a complete nonempty connected layer array. Sizes use
 * the host address domain; buffers remain unchanged during the entire call. */
typedef struct {
    uint32_t version; /* Exact ownership ABI version1. */
    uint32_t flags; /* 1 trace,2 binary drive,4 linear final,8 supplied state. */
    sc_if_layer *layers; /* Live aligned layer descriptors. */
    size_t layer_count; /* Positive layer count. */
    double *frames; /* Time/batch/input frames; null allowed at zero length. */
    size_t frames_len; /* Number of doubles in frames. */
    size_t steps; /* Nonnegative timestep count. */
    size_t batch; /* Nonnegative batch count. */
    size_t max_working_bytes; /* Positive numeric reservation limit, not RSS. */
} sc_if_request;

/* The data pointer may be null only at zero length. Live pinned numeric storage
 * belongs to the opaque result; the caller may mutate doubles but must keep its
 * owner alive throughout access. Metadata and handle remain unchanged. */
typedef struct {
    double *data; /* Borrowed row-major storage, live until supplying owner free. */
    size_t len; /* Number of doubles. */
} sc_if_view;

/* Version1. No library construction or model operation. */
uint32_t sc_if_abi_version(void);
/* Returns0 success,-1 invalid,-2 reservation,-3 overflow,-4 internal failure.
 * Refusal leaves the exclusive live aligned result slot unchanged. */
int32_t sc_if_replay(sc_if_request *request, void **result);
/* Kind0 output/index0,1 final state,2 state trace,3 IF event trace.
 * Refusal returns-1 without modifying view. Handle is live and not concurrently
 * freed; view is exclusive writable aligned metadata. */
int32_t sc_if_buffer(void *handle, uint32_t kind, size_t index, sc_if_view *view);
/* Null is harmless; otherwise exactly-once free of a live owner from this
 * supplying library after all views expire. No concurrent view access/free. */
void sc_if_free(void *handle);
#endif

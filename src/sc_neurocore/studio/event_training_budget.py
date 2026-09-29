# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event tensor input admission

"""Refuse oversized temporal input before file loading or tensor allocation."""

from __future__ import annotations

import os
import sys

from sc_neurocore.studio.event_training_contract import EventTrainingContract

EVENT_INPUT_LIMIT_ENV = "SC_NEUROCORE_STUDIO_EVENT_INPUT_MAX_BYTES"
DEFAULT_EVENT_INPUT_MAX_BYTES = 64 * 1024 * 1024


def admit_event_training_input(
    contract: EventTrainingContract, batch_size: int
) -> dict[str, int | str]:
    """Check the operator's budget for encoded inputs and loader collation.

    Parameters
    ----------
    contract:
        Structurally validated temporal input and complete split plan.
    batch_size:
        Requested positive batch size. Accounting uses the largest actual
        batch possible in the selected training and evaluation parts.

    Returns
    -------
    dict
        Path-free accounting receipt with explicit sample, collation and
        boolean encoding buffers, dimensions and operator limit.

    Raises
    ------
    ValueError
        Batch size cannot be represented by the loader, the operator limit
        is invalid, or the accounted input buffers exceed that limit.

    Notes
    -----
    This is an admission budget for input tensors, not a bound on total
    process memory. Raw recording arrays, model parameters, optimiser state,
    activations and library overhead require the worker's separate resource
    limits. No tensor is allocated to calculate this receipt.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("event batch size must be a positive integer")
    if not 0 < batch_size <= sys.maxsize:
        raise ValueError("event batch size is outside the loader's integer range")
    configured = os.environ.get(EVENT_INPUT_LIMIT_ENV)
    try:
        limit = DEFAULT_EVENT_INPUT_MAX_BYTES if configured is None else int(configured)
    except ValueError as exc:
        raise ValueError(f"operator {EVENT_INPUT_LIMIT_ENV} must be a positive integer") from exc
    if limit <= 0:
        raise ValueError(f"operator {EVENT_INPUT_LIMIT_ENV} must be a positive integer")
    samples = max(
        len(contract.split.assignment[contract.train_split]),
        len(contract.split.assignment[contract.evaluation_split]),
    )
    if samples == 0:
        raise ValueError("event training input parts must not be empty")
    effective_batch = min(batch_size, samples)
    sample_elements = contract.encoder.n_steps * contract.encoder.channels
    sample_tensors = 4 * effective_batch * sample_elements
    collation = sample_tensors
    encoding = sample_elements
    accounted = sample_tensors + collation + encoding
    if accounted > limit:
        raise ValueError(
            f"event input buffers require {accounted} bytes, exceeding the operator "
            f"limit of {limit}; reduce batch_size or timesteps, or ask the operator "
            f"to configure {EVENT_INPUT_LIMIT_ENV}"
        )
    return {
        "schema": "studio.event-input-budget.v1",
        "requested_batch_size": batch_size,
        "effective_batch_size": effective_batch,
        "timesteps": contract.encoder.n_steps,
        "input_channels": contract.encoder.channels,
        "sample_tensors_bytes": sample_tensors,
        "collation_bytes": collation,
        "encoding_buffer_bytes": encoding,
        "accounted_input_bytes": accounted,
        "operator_limit_bytes": limit,
    }

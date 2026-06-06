# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Packaged HDL resources for offline deployment profiles."""

from .aer_priority_queue_reference import AERPriorityEvent, AERPriorityQueueReference
from .resources import baseline_primitive_text, list_baseline_primitive_rtl

__all__ = [
    "AERPriorityEvent",
    "AERPriorityQueueReference",
    "baseline_primitive_text",
    "list_baseline_primitive_rtl",
]

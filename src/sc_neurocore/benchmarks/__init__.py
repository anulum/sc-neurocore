# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroBench-compatible benchmark framework

"""NeuroBench-compatible benchmarking for SC-NeuroCore models."""

from .metrics import compute_metrics, BenchmarkResult
from .tasks import TASKS

__all__ = ["compute_metrics", "BenchmarkResult", "TASKS"]

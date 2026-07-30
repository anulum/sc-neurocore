# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Model 52 benchmark facade over the source/binary-bound runner."""

from benchmarks._non_resetting_lif_benchmark import BenchmarkSpec, REPOSITORY, run

__all__ = ["BenchmarkSpec", "REPOSITORY", "run"]

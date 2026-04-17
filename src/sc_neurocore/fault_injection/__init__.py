# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.fault_injection package public API surface

"""sc_neurocore.fault_injection — Radiation-grade fault injection and resilience testing.

Tier: industrial.

Single module:

- ``fault_injection`` — single-event-upset (SEU) and bit-flip
  injectors with configurable radiation profiles, plus a
  resilience benchmarking harness that quantifies the system's
  output drift as a function of fault rate.
"""

from sc_neurocore.fault_injection.fault_injection import (
    FaultInjectionResult,
    FaultInjector,
    FaultModel,
    RadiationProfile,
    ResilienceBenchmark,
    ResilienceReport,
)

__tier__ = "industrial"

__all__ = [
    # enums
    "FaultModel",
    # dataclasses
    "FaultInjectionResult",
    "RadiationProfile",
    "ResilienceReport",
    # injectors / benchmarks
    "FaultInjector",
    "ResilienceBenchmark",
]

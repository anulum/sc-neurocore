# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SoC and chiplet facade

"""System-on-Chip (SoC) and multi-die chiplet deployment facade."""

from __future__ import annotations

from .bram_array import (
    generate_bram_array,
)
from .cdc_synchroniser import (
    generate_cdc_synchroniser,
)
from .cxl_coherence import (
    CXLMapping,
    advise_cxl_mapping,
)
from .memory_map import (
    MemoryMap,
    generate_memory_map,
)
from .multi_die_floorplan import (
    FloorplanResult,
    plan_multi_die_floorplan,
)
from .pim_layout import (
    PIMLayout,
    plan_pim_layout,
)
from .pipeline_wrapper import (
    generate_pipeline_wrapper,
)
from .power_domain_wrapper import (
    generate_power_domain_wrapper,
)
from .storage_recommendation import (
    StorageRecommendation,
    storage_recommendation,
)
from .tmr_wrapper import (
    generate_tmr_wrapper,
)
from .ucie_partitioning import (
    UCIePartition,
    advise_ucie_partition,
)
from .ucie_protocol_mapper import (
    UCIeMapping,
    map_ucie_protocol,
)

__all__ = [
    "CXLMapping",
    "FloorplanResult",
    "MemoryMap",
    "PIMLayout",
    "StorageRecommendation",
    "UCIeMapping",
    "UCIePartition",
    "advise_cxl_mapping",
    "advise_ucie_partition",
    "generate_bram_array",
    "generate_cdc_synchroniser",
    "generate_memory_map",
    "generate_pipeline_wrapper",
    "generate_power_domain_wrapper",
    "generate_tmr_wrapper",
    "map_ucie_protocol",
    "plan_multi_die_floorplan",
    "plan_pim_layout",
    "storage_recommendation",
]

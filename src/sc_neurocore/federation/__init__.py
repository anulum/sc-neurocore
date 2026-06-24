# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation vertical

"""SC-NeuroCore's STUDIO federation vertical on the scpn-studio-platform contract.

Hub-facing federation surface (schema-A capability manifest + schema-B evidence
bundles), distinct from the local FastAPI Studio web app under
``sc_neurocore.studio``. Importing this package requires the ``federation`` extra
(``pip install sc-neurocore[federation]`` → ``scpn-studio-platform>=0.6,<0.7``); a
clean ``ModuleNotFoundError`` is raised when the platform SDK is absent, which is
why tests guard with ``pytest.importorskip("scpn_studio_platform")``.

Example
-------
>>> from sc_neurocore.federation import build_manifest, fpga_deployment_evidence
>>> manifest = build_manifest()
>>> manifest.studio
'sc-neurocore'
"""

from __future__ import annotations

from .evidence import (
    FpgaDeploymentResult,
    ScInferenceResult,
    fpga_deployment_evidence,
    sc_inference_evidence,
)
from .manifest import (
    PLATFORM_SDK_RANGE,
    PROTOCOL_VERSION,
    STUDIO_VERSION,
    build_manifest,
    declared_surface,
)
from .verbs import (
    NEUROCORE_VERBS,
    STUDIO_ID,
    core_verbs,
    domain_verbs,
    evidence_schemas,
)

__all__ = [
    "NEUROCORE_VERBS",
    "PLATFORM_SDK_RANGE",
    "PROTOCOL_VERSION",
    "STUDIO_ID",
    "STUDIO_VERSION",
    "FpgaDeploymentResult",
    "ScInferenceResult",
    "build_manifest",
    "core_verbs",
    "declared_surface",
    "domain_verbs",
    "evidence_schemas",
    "fpga_deployment_evidence",
    "sc_inference_evidence",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation capability manifest

"""The SC-NeuroCore studio's capability manifest (schema A) on the platform contract.

Authors SC-NeuroCore's :class:`scpn_studio_platform.manifest.CapabilityManifest`:
the verbs it advertises, the evidence schemas they emit, and the content-addressed
digest of that declared surface. The digest is computed with the platform's
:func:`scpn_studio_platform.manifest.content_digest` over the canonical JSON of the
declared verbs, evidence schemas and UI contract, so it is reproducible
across checkouts and independent of git state — the ``capability_manifest`` count-drift gotcha does not
apply here.

This federation manifest is distinct from the local FastAPI Studio web app under
``sc_neurocore.studio``; it is the Hub-facing federation surface, not a UI server.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

from scpn_studio_platform.manifest import (
    CapabilityManifest,
    TransportProfile,
    UiModule,
    content_digest,
)

from .verbs import NEUROCORE_VERBS, STUDIO_ID, evidence_schemas

PLATFORM_SDK_RANGE = ">=0.11,<0.12"
"""The platform SDK SemVer range the studio builds on (matches the ``federation`` extra)."""

PROTOCOL_VERSION = "1"
"""The SYNAPSE wire protocol version the studio pins."""

STUDIO_UI = UiModule(
    remote_entry="https://www.anulum.org/studios/sc-neurocore/remoteEntry.js",
    exposes=("./SnnStudioPanel",),
    federation="module-federation-2",
)
"""The producer-owned remote contract; declaration does not attest host adoption."""


def _resolve_studio_version() -> str:
    """Return the source ``sc-neurocore`` version, or an installed fallback.

    Returns
    -------
    str
        The package source version when importable, the installed distribution
        version otherwise, or ``"0+unknown"`` so the manifest never carries a
        fabricated version.
    """
    try:
        from sc_neurocore import __version__
    except ImportError:  # pragma: no cover - only in a broken source tree
        __version__ = ""
    if __version__:
        return __version__
    try:
        return version("sc-neurocore")
    except PackageNotFoundError:  # pragma: no cover - only in a non-installed tree
        return "0+unknown"


STUDIO_VERSION = _resolve_studio_version()
"""The SC-NeuroCore studio version this manifest stamps (source package version)."""


def declared_surface(*, ui_module: UiModule = STUDIO_UI) -> dict[str, bytes]:
    """Return the content-addressable declared surface of the SC-NeuroCore studio.

    The surface is the canonical JSON of each advertised verb, the evidence
    schema list and the UI module, keyed by a stable logical path. Hashing surface
    *content* (not git state) makes the digest reproducible across checkouts.

    Parameters
    ----------
    ui_module
        Producer-selected UI descriptor. Local entry paths and integrity pins
        are included in the declared digest, just like the published URL.

    Returns
    -------
    dict[str, bytes]
        Mapping of logical path to canonical-JSON bytes, suitable for
        :func:`scpn_studio_platform.manifest.content_digest`.
    """
    surface: dict[str, bytes] = {
        f"verb/{verb.name}": json.dumps(
            verb.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        for verb in NEUROCORE_VERBS
    }
    surface["evidence/schemas"] = json.dumps(
        list(evidence_schemas()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    surface["ui/module"] = json.dumps(
        ui_module.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return surface


def build_manifest(
    *, studio_version: str = STUDIO_VERSION, ui_module: UiModule = STUDIO_UI
) -> CapabilityManifest:
    """Build the SC-NeuroCore studio's capability manifest.

    Parameters
    ----------
    studio_version
        The studio version to stamp; defaults to :data:`STUDIO_VERSION` (the
        source ``sc-neurocore`` version).
    ui_module
        Producer-selected UI descriptor. Defaults to the published remote;
        its full wire representation contributes to the content digest.

    Returns
    -------
    CapabilityManifest
        The schema-A manifest, with a content digest over :func:`declared_surface`.
    """
    return CapabilityManifest(
        contract_era="v2",
        studio=STUDIO_ID,
        studio_version=studio_version,
        platform_sdk=PLATFORM_SDK_RANGE,
        content_digest=content_digest(declared_surface(ui_module=ui_module)),
        protocol_version=PROTOCOL_VERSION,
        transport_profile=TransportProfile.LOCAL_FIRST,
        verbs=NEUROCORE_VERBS,
        evidence_types=evidence_schemas(),
        ui_module=ui_module,
    )

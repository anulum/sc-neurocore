# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue receipt snapshots and verified detail

"""Share parsed receipt inputs within one operation while verifying every model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from sc_neurocore.neurons.facet_receipts import RECEIPT_DIR, FacetReceipt, latest_receipts
from sc_neurocore.neurons.readiness import verify_model

ReceiptIndex = Mapping[tuple[str, str, str], tuple[Path, FacetReceipt]]


class ReceiptSnapshot:
    """Load one receipt inventory lazily for one catalogue, detail or seal operation."""

    def __init__(self, directory: Path = RECEIPT_DIR) -> None:
        """Select an input directory without reading or verifying any receipt.

        Parameters
        ----------
        directory : pathlib.Path
            Actual append-only facet receipt store for this operation.
        """
        self._directory = directory
        self._receipts: ReceiptIndex | None = None

    def read(self) -> ReceiptIndex:
        """Read the inventory once, retrying normally after a failed parse.

        Returns
        -------
        mapping
            Newest parsed receipt for each exact class, facet and profile.

        Raises
        ------
        FacetReceiptError
            A malformed receipt prevents a complete inventory from being read.
        OSError
            An input cannot be read.
        """
        if self._receipts is None:
            self._receipts = latest_receipts(self._directory)
        return self._receipts


def receipt_verified_detail(
    class_name: str, *, profile: str | None, snapshot: ReceiptSnapshot | None = None
) -> dict[str, object]:
    """Verify a model against receipt inputs and retain all per-facet diagnostics.

    Parameters
    ----------
    class_name : str
        Registered canonical class name.
    profile : str or None
        Exact Studio schema profile, or the verifier's normal profile selection.
    snapshot : ReceiptSnapshot or None
        Operation-local parsed inputs; an independent call reads fresh inputs.

    Returns
    -------
    dict
        Verified profile, science/silicon tiers and every facet's status,
        receipt, changed subjects, problems and evidence references.
    """
    record = verify_model(
        class_name, profile=profile, receipts=None if snapshot is None else snapshot.read()
    )
    return {
        "profile": record.profile,
        "science_tier": record.verified_science,
        "science_label": record.verified_science_label,
        "silicon_tier": record.verified_silicon,
        "silicon_label": record.verified_silicon_label,
        "facets": [
            {
                "facet": facet.facet,
                "declared": facet.declared,
                "status": facet.status,
                "receipt": facet.receipt,
                "changed_subjects": list(facet.changed_subjects),
                "problems": list(facet.problems),
                "evidence": [reference.to_public_dict() for reference in facet.evidence],
            }
            for facet in record.facets
        ],
    }

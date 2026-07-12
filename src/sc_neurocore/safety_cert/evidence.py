# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — safety evidence manifests

"""Hash and verify explicit evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sc_neurocore.safety_cert.certification import CertificationPackage

_CATEGORIES = frozenset(
    {
        "formal",
        "test",
        "analysis",
        "design",
        "report",
        "hil",
        "hardware-in-loop",
        "hardware_in_loop",
        "security",
        "timing",
        "latency",
        "formal_timing",
    }
)
_PACKAGE_CATEGORIES = {
    "traceability_matrix.md": "report",
    "fmeda_report.md": "analysis",
    "formal_proof_cert.md": "formal",
    "wcet_analysis.md": "timing",
    "compliance_checklist.md": "report",
}
_PACKAGE_DESCRIPTIONS = {
    "traceability_matrix.md": "Requirement traceability",
    "fmeda_report.md": "FMEDA analysis",
    "formal_proof_cert.md": "Formal-property evidence",
    "wcet_analysis.md": "WCET analysis",
    "compliance_checklist.md": "Compliance checklist",
}


@dataclass
class EvidenceItem:
    """One relative evidence filename with optional full SHA-256 digest."""

    filename: str
    category: str
    description: str
    sha256: str = ""

    def __post_init__(self) -> None:
        """Validate metadata and reject unsafe or ambiguous paths."""
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        candidate = PurePosixPath(self.filename)
        if (
            candidate.is_absolute()
            or self.filename != candidate.as_posix()
            or "\\" in self.filename
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("filename must be a normalised relative POSIX path")
        if not isinstance(self.category, str) or self.category not in _CATEGORIES:
            choices = ", ".join(sorted(_CATEGORIES))
            raise ValueError(f"category must be one of: {choices}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.sha256, str):
            raise ValueError("sha256 must be a string")
        if self.sha256:
            if len(self.sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in self.sha256
            ):
                raise ValueError("sha256 must be a 64-character hexadecimal digest when provided")
            self.sha256 = self.sha256.lower()


class EvidenceBag:
    """Ordered manifest of uniquely named evidence artifacts."""

    def __init__(self) -> None:
        """Create an empty evidence manifest."""
        self.items: list[EvidenceItem] = []

    def _validated_items(self) -> list[EvidenceItem]:
        for item in self.items:
            if not isinstance(item, EvidenceItem):
                raise ValueError("items must contain EvidenceItem entries")
        filenames = [item.filename for item in self.items]
        if len(filenames) != len(set(filenames)):
            raise ValueError("evidence filenames must be unique")
        return self.items

    def add(self, item: EvidenceItem) -> None:
        """Add one item while preserving filename uniqueness."""
        if not isinstance(item, EvidenceItem):
            raise ValueError("item must be an EvidenceItem")
        if any(existing.filename == item.filename for existing in self.items):
            raise ValueError("evidence filenames must be unique")
        self.items.append(item)

    def add_from_package(self, package: CertificationPackage) -> None:
        """Index every in-memory package report with its real content digest."""
        if not isinstance(package, CertificationPackage):
            raise ValueError("pkg must be a CertificationPackage")
        package.checklist_report()
        for filename, content in package.artifacts().items():
            self.add(
                EvidenceItem(
                    filename=filename,
                    category=_PACKAGE_CATEGORIES[filename],
                    description=_PACKAGE_DESCRIPTIONS[filename],
                    sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                )
            )

    @property
    def file_count(self) -> int:
        """Return the number of validated manifest rows."""
        return len(self._validated_items())

    def manifest(self) -> str:
        """Render a Markdown manifest including declared digests."""
        lines = ["# Evidence Bag Manifest", f"Items: {self.file_count}", ""]
        for item in self._validated_items():
            digest = item.sha256 or "UNHASHED"
            lines.append(
                f"- [{item.category}] {item.filename}: {item.description} (sha256: {digest})"
            )
        return "\n".join(lines)

    def content_sha256(self) -> str:
        """Return the full digest of all canonical manifest fields."""
        payload = [
            {
                "category": item.category,
                "description": item.description,
                "filename": item.filename,
                "sha256": item.sha256,
            }
            for item in sorted(self._validated_items(), key=lambda candidate: candidate.filename)
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compute_hashes(self) -> str:
        """Return the historical 32-character prefix of the manifest digest."""
        return self.content_sha256()[:32]

    def verify(self, directory: str | Path) -> bool:
        """Verify all declared digests against regular non-symlink files."""
        if not isinstance(directory, str | Path):
            raise ValueError("directory must be a string or Path")
        root = Path(directory)
        if not root.is_dir() or root.is_symlink():
            return False
        for item in self._validated_items():
            if not item.sha256:
                return False
            candidate = root.joinpath(*PurePosixPath(item.filename).parts)
            if candidate.is_symlink() or not candidate.is_file():
                return False
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != item.sha256:
                return False
        return True


__all__ = [
    "EvidenceItem",
    "EvidenceBag",
]

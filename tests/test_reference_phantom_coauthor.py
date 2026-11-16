# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — gate: no new citation of the recurring false co-author "Bhatt"

"""Count every remaining "Bhatt" citation per file, so a new one fails.

Generated model pages and docstrings repeatedly cited a false co-author,
"Bhatt", in place of the real authors (``Bhatt, D. & Bhatt, E.``, ``Bhatt DL``).
The citations whose registry record was found were corrected; the ones that
no registry holds stay as written until they are reviewed, and a few lines
record earlier corrections. This gate pins how many of those remain in each
file: a citation added anywhere fails it, and a reviewed one lowers the pinned
count. The ``KilincBhattMapNeuron`` compatibility alias is a code name, not a
citation, and is not counted; neither is a real author whose surname only
begins with the letters (Bhattacharyya).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("docs", "engine/src", "src", "bridge", "CHANGELOG.md", "README.md")
SUFFIXES = frozenset({".md", ".rs", ".py", ".toml", ".json", ".jl", ".go", ".mojo"})
ALIAS = re.compile(r"[Kk]ilinc[\s_&-]*[Bb]hatt\w*|KilincBhatt\w*")
SURNAME = re.compile(r"\bBhatt\b")

# Citations kept for review, and lines that record earlier corrections.
PINNED = {
    "CHANGELOG.md": 5,
    "docs/API_REFERENCE.md": 1,
    "docs/CHANGELOG.md": 5,
    "docs/api/models/ai_optimized.md": 1,
    "docs/api/models/astrocyte_adapter.md": 1,
    "docs/api/models/atype_k_neuron.md": 1,
    "docs/api/models/bk_neuron.md": 6,
    "docs/api/models/elboustani_network.md": 1,
    "docs/api/models/endocrine_beta_cell.md": 2,
    "docs/api/models/graded_synapse_neuron.md": 4,
    "docs/api/models/ih_neuron.md": 1,
    "docs/api/models/kilinc_bhatt_map.md": 1,
    "docs/api/models/loihi_cuba.md": 1,
    "docs/api/models/ltc.md": 1,
    "docs/api/models/merkel_cell.md": 1,
    "docs/api/models/myelinated_axon.md": 1,
    "docs/api/models/olfactory_receptor.md": 2,
    "docs/api/models/outer_hair_cell.md": 1,
    "docs/api/models/smooth_muscle_cell.md": 3,
    "docs/api/models/tum_network.md": 1,
    "docs/api/models/unipolar_brush_cell.md": 8,
    "docs/api/models/vip_neuron.md": 1,
    "docs/tutorials/67_spike_dsp.md": 1,
    "docs/validation/adaptive_threshold_if_source_fidelity.md": 1,
    "engine/src/neurons/cerebellar/unipolar_brush.rs": 1,
    "engine/src/neurons/channels/bk.rs": 2,
    "engine/src/neurons/interneurons/vip_neuron.rs": 1,
    "engine/src/neurons/population/el_boustani_network.rs": 1,
    "src/sc_neurocore/neurons/models/vip_neuron.py": 1,
}


def count_citations(text: str) -> int:
    """Return the number of standalone "Bhatt" words, with the alias names removed."""
    return sum(len(SURNAME.findall(ALIAS.sub("", line))) for line in text.splitlines())


def scanned_files(root: Path) -> list[Path]:
    """Return every scanned text file below ``root``, leaving out internal notes."""
    files: list[Path] = []
    for entry in SCANNED:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix in SUFFIXES and "internal" not in p.parts
            )
    return files


def citation_counts(root: Path) -> dict[str, int]:
    """Map each scanned file that cites the surname to its count."""
    counts: dict[str, int] = {}
    for path in scanned_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = count_citations(text)
        if found:
            counts[path.relative_to(root).as_posix()] = found
    return counts


def test_remaining_citations_match_the_pinned_counts() -> None:
    """Every file's count equals its pin: no new citation, and a reviewed one lowers the pin."""
    counts = citation_counts(REPO_ROOT)
    added = {path: n for path, n in counts.items() if n > PINNED.get(path, 0)}
    lowered = {path: counts.get(path, 0) for path, n in PINNED.items() if counts.get(path, 0) < n}
    assert not added, f"new 'Bhatt' citations (check the source, then correct): {added}"
    assert not lowered, f"fewer 'Bhatt' citations than pinned (lower PINNED): {lowered}"


def test_count_ignores_the_alias_and_longer_surnames() -> None:
    """The alias names and Bhattacharyya are not counted; a citation in a line with the alias is."""
    assert count_citations("KilincBhattMapNeuron kilinc_bhatt_map Kilinc & Bhatt") == 0
    assert count_citations("Nunzi, M. G., Birnstiel, S., Bhattacharyya, B. J.") == 0
    assert count_citations("Bhatt DL, Storm JF (2003)\nKilincBhattMapNeuron, or Bhatt.") == 2


def test_a_new_citation_in_a_scanned_tree_is_counted(tmp_path: Path) -> None:
    """A page added under a scanned folder is found, and an internal note is not."""
    pages = tmp_path / "docs" / "api" / "models"
    pages.mkdir(parents=True)
    (pages / "new_model.md").write_text("1. Smith, A. & Bhatt, D. (2020).\n", encoding="utf-8")
    internal = tmp_path / "docs" / "internal"
    internal.mkdir()
    (internal / "notes.md").write_text("Bhatt\n", encoding="utf-8")
    assert citation_counts(tmp_path) == {"docs/api/models/new_model.md": 1}

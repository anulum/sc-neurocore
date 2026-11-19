# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — registry-proven citation corrections on the model pages

"""Hold the citation corrections that two registries prove, and the one form kept for review.

Each corrected entry is its work's record in Crossref, read a second time by DOI in PubMed or Semantic Scholar.
The record identifies the work through the first author with the title opening, or with the journal and the
year, volume or first page; the remaining fields then come from it. A wrong form must not return in the
documentation or the engine sources. One form stays where it is, because its own description points to another
work than its journal and page do; it is pinned here so that it cannot change unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.benchmark_evidence_gate import tracked_files

ROOT = Path(__file__).resolve().parents[1]

#: (wrong form, corrected form, files that carry the corrected form)
CORRECTIONS = [
    (
        "Proc IEEE 111:1016, 2021",
        "Proc IEEE 111:1016, 2023",
        ["docs/api/models/klif.md"],
    ),
    (
        "J Neurosci 28:11003, 2008",
        "Pillow et al., J Neurosci 25:11003, 2005",
        ["docs/api/models/retinal_ganglion.md"],
    ),
    (
        "Neuron* 65:801–813, 2010",
        "Vervaeke et al., *Neuron* 67:435–451, 2010",
        ["docs/api/models/gap_junction_neuron.md"],
    ),
    (
        "Neuron 65:801, 2010",
        "Vervaeke et al., Neuron 67:435, 2010",
        ["engine/src/neurons/misc/gap_junction.rs"],
    ),
    (
        "J Physiol* 531:567–584, 2001",
        "Hirst & Edwards, *J Physiol* 535:165–180, 2001",
        ["docs/api/models/smooth_muscle_cell.md"],
    ),
    (
        "J Physiol 531:567, 2001",
        "Hirst & Edwards, J Physiol 535:165, 2001",
        ["engine/src/neurons/misc/smooth_muscle.rs"],
    ),
    ("20(RC50), 1–5.", "20(1), RC50.", ["docs/api/models/tum_network.md"]),
    (
        "Adv Exp Med Biol* 723:191–197",
        "*Prog Retin Eye Res* 31(5):442–466",
        ["docs/api/models/rod_photoreceptor.md"],
    ),
    (
        "119(1), 416–432",
        "7(3), 218–235",
        ["docs/api/models/inner_hair_cell.md"],
    ),
    ("10.1121/1.2133496", "10.1007/s10162-006-0037-8", ["docs/api/models/inner_hair_cell.md"]),
]

#: Kept as written and listed for review: the description "(coupled GLM)" names Pillow et al. 2008
#: (Nature 454:995), while the journal and page name Pillow et al. 2005 (J Neurosci 25:11003).
KEPT_FOR_REVIEW = {
    "engine/src/neurons/sensory/retinal_ganglion_cell.rs": "Pillow et al., J Neurosci 28:11003, 2008 (coupled GLM).",
}


#: (directory, suffix) pairs whose files must not carry a wrong form.
SOURCE_KINDS = [("docs", ".md"), ("engine/src", ".rs"), ("src", ".py"), ("src", ".toml")]


def _sources() -> list[Path]:
    """Return the repository's files of each source kind, without the changelogs.

    In a Git work tree these are the files Git tracks, so ignored material (internal notes that quote the old
    forms, downloaded papers, a virtual environment or a Mojo toolchain inside the package) is never read.
    Outside a work tree every file is walked.
    """
    paths: list[Path] = []
    for directory, suffix in SOURCE_KINDS:
        tracked = tracked_files(ROOT, directory)
        if tracked is None:
            paths.extend((ROOT / directory).rglob(f"*{suffix}"))
        else:
            paths.extend(ROOT / name for name in tracked if name.endswith(suffix))
    return [path for path in paths if "CHANGELOG" not in path.name]


def _texts_without_the_kept_forms() -> dict[str, str]:
    texts = {}
    for path in _sources():
        name = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8", errors="replace")
        texts[name] = text.replace(KEPT_FOR_REVIEW.get(name, "\0"), "")
    return texts


@pytest.mark.parametrize(("wrong", "right", "files"), CORRECTIONS)
def test_each_correction_stands_and_its_wrong_form_is_gone(
    wrong: str, right: str, files: list[str]
) -> None:
    """The corrected form stands in its files, and the wrong form returns nowhere."""
    for name in files:
        assert right in (ROOT / name).read_text(encoding="utf-8"), name
    carrying = sorted(
        name for name, text in _texts_without_the_kept_forms().items() if wrong in text
    )
    assert carrying == [], carrying


def test_the_form_kept_for_review_is_where_it_was() -> None:
    """The one ambiguous form is kept as written until a review decides which work it means."""
    for name, form in KEPT_FOR_REVIEW.items():
        assert (ROOT / name).read_text(encoding="utf-8").count(form) == 1, name


def test_the_scan_reads_tracked_sources_only() -> None:
    """Every scanned file is tracked, and the corrected model pages are among them."""
    names = {path.relative_to(ROOT).as_posix() for path in _sources()}
    corrected = {name for _, _, files in CORRECTIONS for name in files}
    assert corrected <= names
    tracked = tracked_files(ROOT, ".")
    if tracked is not None:
        assert names <= set(tracked)

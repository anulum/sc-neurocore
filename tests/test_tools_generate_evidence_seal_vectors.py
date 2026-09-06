# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The generator behind the cross-runtime seal corpus

"""The corpus is only evidence while the tool that writes it is trustworthy.

Two suites in two runtimes read one file and agree. That agreement is worth
what the file is worth: a draw that quietly returned the same handful of
values, or a ``--check`` that reported freshness without comparing anything,
would leave the parity claim resting on nothing.

These cases hold the tool to what the corpus needs: a reproducible draw across
the whole exponent range, a rendering the repository can store byte for byte,
and a freshness check that actually fails on a stale file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.evidence_seal import EVIDENCE_SEAL_SCHEMA_VERSION
from tools.generate_evidence_seal_vectors import (
    CORPUS_PATH,
    CORPUS_SEED,
    CORPUS_SIZE,
    build_corpus,
    main,
    random_doubles,
    render_corpus,
)


class TestTheDraw:
    def test_it_is_reproducible(self) -> None:
        """A corpus nobody can redraw cannot be reviewed."""
        assert random_doubles(seed=7, count=32) == random_doubles(seed=7, count=32)

    def test_a_different_seed_draws_different_doubles(self) -> None:
        assert random_doubles(seed=7, count=32) != random_doubles(seed=8, count=32)

    def test_it_returns_the_count_asked_for(self) -> None:
        assert len(random_doubles(seed=CORPUS_SEED, count=5)) == 5

    def test_every_drawn_double_is_finite(self) -> None:
        """Non-finite doubles are refused by the seal and belong to its own cases."""
        for value in random_doubles(seed=3, count=256):
            assert value == value
            assert abs(value) != float("inf")

    def test_the_draw_is_over_bit_patterns_not_magnitudes(self) -> None:
        """A draw over magnitudes would never reach the exponents that diverge."""
        magnitudes = [abs(value) for value in random_doubles(seed=3, count=256) if value != 0.0]

        assert min(magnitudes) < 1e-100
        assert max(magnitudes) > 1e100


class TestTheDocument:
    def test_it_states_the_contract_the_seal_implements(self) -> None:
        corpus = build_corpus(seed=CORPUS_SEED, count=8)

        assert corpus["schema_version"] == EVIDENCE_SEAL_SCHEMA_VERSION
        assert corpus["seed"] == CORPUS_SEED
        assert corpus["count"] == 8
        assert len(corpus["vectors"]) == 8

    def test_every_vector_carries_a_value_and_its_canonical_text(self) -> None:
        for vector in build_corpus(seed=CORPUS_SEED, count=8)["vectors"]:
            assert set(vector) == {"canonical", "value"}
            assert float(vector["canonical"]) == vector["value"]

    def test_the_rendering_is_stable_and_ends_with_a_newline(self) -> None:
        """The repository stores this text, so it has to be byte-reproducible."""
        text = render_corpus(build_corpus(seed=CORPUS_SEED, count=8))

        assert text == render_corpus(build_corpus(seed=CORPUS_SEED, count=8))
        assert text.endswith("\n")
        assert json.loads(text)["count"] == 8


class TestTheCommandLine:
    def test_it_writes_the_corpus_where_it_is_told(self, tmp_path: Path) -> None:
        target = tmp_path / "corpus.json"

        assert main(["--output", str(target)]) == 0
        assert target.read_text(encoding="utf-8") == render_corpus(build_corpus())

    def test_check_passes_on_the_corpus_the_repository_stores(self) -> None:
        """The committed file is the one this build writes."""
        assert main(["--check"]) == 0
        assert CORPUS_PATH.is_file()

    def test_check_fails_on_a_stale_file_without_rewriting_it(self, tmp_path: Path) -> None:
        stale = tmp_path / "corpus.json"
        stale.write_text('{"count": 0}\n', encoding="utf-8")

        assert main(["--check", "--output", str(stale)]) == 1
        assert stale.read_text(encoding="utf-8") == '{"count": 0}\n'

    def test_check_fails_when_there_is_no_file_at_all(self, tmp_path: Path) -> None:
        assert main(["--check", "--output", str(tmp_path / "absent.json")]) == 1

    def test_the_default_size_is_the_one_both_suites_expect(self) -> None:
        assert json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["count"] == CORPUS_SIZE

    def test_it_reports_what_it_wrote(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "corpus.json"
        main(["--output", str(target)])

        assert str(target) in capsys.readouterr().out

    def test_it_runs_as_a_command_and_reports_a_stale_corpus(self, tmp_path: Path) -> None:
        """It is used from a shell, so the entry point has to work from one."""
        stale = tmp_path / "corpus.json"
        stale.write_text("{}\n", encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "tools/generate_evidence_seal_vectors.py"

        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(script), "--check", "--output", str(stale)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        assert completed.returncode == 1
        assert "stale evidence seal corpus" in completed.stdout

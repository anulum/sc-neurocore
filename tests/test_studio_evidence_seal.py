# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio cross-runtime evidence seal

"""A seal that survives the browser, and refuses what would not.

The defect these cases hold closed was measured on the real surface: a model run
of ``AdExNeuron`` recorded ``result_sha256`` ``771d8e51…`` and the identical
payload, after the JSON round trip every exported artefact makes through the
operator's browser, re-sealed to ``813d3005…``. The values had not changed —
``1.0`` had come back as ``1``. No verifier could exist while that was true.

**What these cases can and cannot show.** Python cannot perform JavaScript's
JSON round trip, so nothing here simulates one: an earlier helper claimed to
and did not, because ``json.loads(json.dumps(1.0))`` is ``1.0`` in Python and
never the ``1`` a browser returns. The cross-runtime claim is carried by two
committed corpora — the hand-written vectors and a random-double corpus — read
by **both** suites. Each runtime parses the same JSON with its own parser and
must reach the same canonical text; ``evidenceSeal.test.ts`` additionally puts
every vector through a real ``JSON.parse(JSON.stringify(...))`` before sealing
it, which is the browser round trip itself, in the runtime that has one. What
these cases show is this side: the canonical form, the refusals, and that the
committed corpora still describe this build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.evidence_seal import (
    EVIDENCE_SEAL_ALGORITHM,
    EVIDENCE_SEAL_SCHEMA_VERSION,
    EvidenceSealError,
    canonical_seal_text,
    seal_sha256,
)
from tools.generate_evidence_seal_vectors import CORPUS_SIZE, build_corpus, render_corpus

#: Vectors shared with ``evidenceSeal.test.ts``. One file, both runtimes: a
#: divergence fails on whichever side drifted.
_VECTORS = Path(__file__).resolve().parents[1] / "studio/frontend/src/evidenceSealVectors.json"
#: The random-double corpus both suites read; see
#: ``tools/generate_evidence_seal_vectors.py``.
_CORPUS = Path(__file__).resolve().parents[1] / "studio/frontend/src/evidenceSealRandomVectors.json"


class TestCanonicalForm:
    def test_an_integral_float_and_its_integer_seal_alike(self) -> None:
        """The browser returns 1.0 as 1; both must mean the same run."""
        assert canonical_seal_text(1.0) == canonical_seal_text(1) == "1"
        assert canonical_seal_text(-70.0) == "-70"

    def test_negative_zero_is_zero(self) -> None:
        assert canonical_seal_text(-0.0) == canonical_seal_text(0.0) == "0"

    def test_a_fraction_keeps_every_digit_that_round_trips(self) -> None:
        assert canonical_seal_text(0.30000000000000004) == "3.0000000000000004e-1"
        assert canonical_seal_text(0.1) == "1e-1"

    def test_exponent_notation_does_not_depend_on_the_runtime_that_wrote_it(self) -> None:
        """Python writes 1e-07 and the browser 1e-7 for one double."""
        assert canonical_seal_text(1e-07) == canonical_seal_text(float("1e-7")) == "1e-7"

    def test_a_large_integral_double_is_written_in_full(self) -> None:
        """Exponent notation would differ between the runtimes above 1e21."""
        assert canonical_seal_text(1e21) == "1" + "0" * 21

    def test_object_keys_are_ordered_by_code_point(self) -> None:
        text = canonical_seal_text({"z": 1, "a": 2, "\U0001f600": 3, "￿": 4})

        assert text.index('"a"') < text.index('"z"') < text.index('"￿"')
        assert text.index('"￿"') < text.index('"\U0001f600"')

    def test_there_is_no_insignificant_whitespace(self) -> None:
        assert canonical_seal_text({"a": [1, 2], "b": {"c": None}}) == '{"a":[1,2],"b":{"c":null}}'

    def test_strings_carry_the_minimal_escapes(self) -> None:
        assert canonical_seal_text('a"b\\c\n\t') == '"a\\"b\\\\c\\n\\t"'
        assert canonical_seal_text("\x01") == '"\\u0001"'

    def test_non_ascii_text_is_written_literally(self) -> None:
        assert canonical_seal_text("ä☃") == '"ä☃"'

    def test_the_digest_is_of_the_canonical_text(self) -> None:
        import hashlib

        value = {"dt": 1.0, "spikes": [0.5]}
        expected = hashlib.sha256(canonical_seal_text(value).encode("utf-8")).hexdigest()

        assert seal_sha256(value) == expected
        assert EVIDENCE_SEAL_ALGORITHM == "sha256"
        assert EVIDENCE_SEAL_SCHEMA_VERSION == "studio.evidence-seal.v1"


class TestTheMeasuredPayload:
    """The shape the defect was measured in, pinned to a recorded digest."""

    #: The payload of the measured ``AdExNeuron`` run, in the shape the
    #: divergence was found in: an integral float the browser returns as an
    #: integer, a fraction, a negative integral float and a small exponent.
    PAYLOAD = {
        "dt": 1.0,
        "n_steps": 100,
        "spike_count": 3.0,
        "states": {"v": [0.1, -70.0, 1e-07]},
    }

    def test_it_seals_to_the_canonical_text_both_runtimes_write(self) -> None:
        """No rendering of Python's own is allowed to reach the digest."""
        assert canonical_seal_text(self.PAYLOAD) == (
            '{"dt":1,"n_steps":100,"spike_count":3,"states":{"v":[1e-1,-70,1e-7]}}'
        )

    def test_this_side_is_stable_across_its_own_serialisation(self) -> None:
        """A weaker claim than the browser's, and the only one Python can make.

        Python's own round trip returns ``1.0`` as ``1.0``, so this shows the
        seal is stable under serialisation on this side. The browser's round
        trip is exercised in ``evidenceSeal.test.ts``, which has one.
        """
        restored = json.loads(json.dumps(self.PAYLOAD))

        assert seal_sha256(restored) == seal_sha256(self.PAYLOAD)

    def test_an_edited_value_seals_differently(self) -> None:
        """Tolerating a runtime's rendering must not tolerate an edit."""
        edited = {**self.PAYLOAD, "spike_count": 4.0}

        assert seal_sha256(edited) != seal_sha256(self.PAYLOAD)


class TestTheRandomCorpus:
    """Bit patterns nobody would think to write down, checked every run."""

    def _document(self) -> dict[str, Any]:
        document: dict[str, Any] = json.loads(_CORPUS.read_text(encoding="utf-8"))
        return document

    def test_it_describes_this_build(self) -> None:
        """A canonical form that changed would leave the corpus stale.

        The corpus is a contract, not a snapshot to refresh: changing the
        canonical text changes every digest already issued under this schema
        version, so a stale corpus must be a failure and not a regeneration.
        """
        document = self._document()

        assert document["schema_version"] == EVIDENCE_SEAL_SCHEMA_VERSION
        assert document["count"] == len(document["vectors"]) == CORPUS_SIZE
        for vector in document["vectors"]:
            assert canonical_seal_text(vector["value"]) == vector["canonical"], vector

    def test_the_committed_file_is_what_the_generator_writes(self) -> None:
        assert _CORPUS.read_text(encoding="utf-8") == render_corpus(build_corpus())

    def test_the_draw_reaches_the_whole_exponent_range(self) -> None:
        """A draw over magnitudes would never reach these, and they diverge."""
        values = [vector["value"] for vector in self._document()["vectors"]]
        magnitudes = [abs(value) for value in values if value != 0.0]

        assert min(magnitudes) < 1e-100
        assert max(magnitudes) > 1e100
        assert any(value < 0 for value in values)

    def test_every_recorded_text_parses_back_to_the_same_double(self) -> None:
        """The canonical text is the value, not an approximation of it."""
        for vector in self._document()["vectors"]:
            assert float(vector["canonical"]) == vector["value"], vector


class TestRefusals:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_number_is_refused(self, value: float) -> None:
        with pytest.raises(EvidenceSealError, match="sealable JSON number"):
            canonical_seal_text(value)

    def test_an_integer_no_double_holds_is_refused(self) -> None:
        """It would come back from the browser as a different number."""
        with pytest.raises(EvidenceSealError, match="not carried exactly"):
            canonical_seal_text(2**53 + 1)

    def test_an_integer_beyond_every_double_is_refused(self) -> None:
        with pytest.raises(EvidenceSealError, match="not carried exactly"):
            canonical_seal_text(10**400)

    def test_a_large_integer_a_double_holds_exactly_is_kept(self) -> None:
        """Magnitude alone does not decide it; factors of two do."""
        assert canonical_seal_text(10**20) == "1" + "0" * 20

    def test_an_unpaired_surrogate_is_refused(self) -> None:
        with pytest.raises(EvidenceSealError, match="unpaired surrogate"):
            canonical_seal_text("\ud800")

    def test_a_non_string_object_key_is_refused(self) -> None:
        with pytest.raises(EvidenceSealError, match="object key must be a string"):
            canonical_seal_text({1: "one"})

    @pytest.mark.parametrize("value", [b"bytes", bytearray(b"bytes"), {1, 2}, object()])
    def test_a_value_that_is_not_json_shaped_is_refused(self, value: object) -> None:
        with pytest.raises(EvidenceSealError, match="not a sealable JSON value"):
            canonical_seal_text(value)


class TestSharedVectors:
    def test_the_committed_vectors_match_this_implementation(self) -> None:
        """The file the browser half reads is regenerated from this side.

        A change to the canonical form that is not reflected in the vectors
        would leave the two runtimes disagreeing silently.
        """
        document = json.loads(_VECTORS.read_text(encoding="utf-8"))

        assert document["schema_version"] == EVIDENCE_SEAL_SCHEMA_VERSION
        assert document["vectors"]
        for vector in document["vectors"]:
            assert canonical_seal_text(vector["value"]) == vector["canonical"], vector

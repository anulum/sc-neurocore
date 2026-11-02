# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Runtime state transport ceiling contracts

"""Check state transport ceilings against measured lane accounting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool() -> Any:
    """Load the conformance tool the way the repository's other tool tests do."""
    path = REPO_ROOT / "tools" / "runtime_state_conformance.py"
    spec = importlib.util.spec_from_file_location("runtime_state_conformance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matrix(dropped: int, complete: int) -> dict[str, Any]:
    """Return the dropped-variable and complete-model counts for one lane."""
    return {"summary": {"per_lane": {"rust-batch": {"dropped": dropped, "complete": complete}}}}


def test_a_lane_at_its_ceiling_passes() -> None:
    """Accept unchanged dropped-variable and complete-model counts."""
    tool = _tool()
    ceiling = {"lanes": {"rust-batch": {"max_dropped": 415, "min_complete": 20}}}

    assert tool.ceiling_verdicts(_matrix(415, 20), ceiling) == []


def test_more_dropped_state_is_a_regression() -> None:
    """The matrix refuses to drift, but regenerating accepts a worse number."""
    tool = _tool()
    ceiling = {"lanes": {"rust-batch": {"max_dropped": 415, "min_complete": 20}}}

    verdicts = tool.ceiling_verdicts(_matrix(416, 20), ceiling)

    assert len(verdicts) == 1
    assert "416 state variables dropped" in verdicts[0]


def test_fewer_complete_models_is_a_regression() -> None:
    """A model that stops being fully carried is a loss even if the count fell."""
    tool = _tool()
    ceiling = {"lanes": {"rust-batch": {"max_dropped": 415, "min_complete": 20}}}

    verdicts = tool.ceiling_verdicts(_matrix(400, 19), ceiling)

    assert len(verdicts) == 1
    assert "19 models fully carried" in verdicts[0]


def test_improvement_passes_in_both_directions() -> None:
    """Transport getting better is the point; only worsening is refused."""
    tool = _tool()
    ceiling = {"lanes": {"rust-batch": {"max_dropped": 415, "min_complete": 20}}}

    assert tool.ceiling_verdicts(_matrix(300, 40), ceiling) == []


def test_a_lane_with_no_ceiling_is_named_rather_than_passed() -> None:
    """A new lane must be recorded deliberately, not admitted by silence."""
    tool = _tool()

    verdicts = tool.ceiling_verdicts(_matrix(10, 1), {"lanes": {}})

    assert len(verdicts) == 1
    assert "no ceiling recorded" in verdicts[0]


def test_the_committed_ceiling_matches_the_live_matrix() -> None:
    """The recorded ceiling is measured, never copied from an expectation."""
    tool = _tool()
    ceiling_path = REPO_ROOT / "tools" / "runtime_state_ceiling.toml"

    ceiling = tool.tomllib.loads(ceiling_path.read_text(encoding="utf-8"))

    assert tool.ceiling_verdicts(tool.build_matrix(), ceiling) == []


def test_a_malformed_lane_table_is_refused() -> None:
    """Reject a ceiling document whose lane limits cannot be interpreted."""
    tool = _tool()

    with pytest.raises(tool.RuntimeStateCeilingError, match="must be a table"):
        tool.ceiling_verdicts(_matrix(10, 1), {"lanes": []})


def test_checking_an_absent_ceiling_fails_without_creating_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require an explicit ceiling update before admitting a missing file."""
    tool = _tool()
    destination = tmp_path / "ceiling.toml"

    assert tool.main(["--check-ceiling", "--ceiling", str(destination)]) == 1
    assert "is absent; run --update-ceiling" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["--check-ceiling", "--update-ceiling"])
def test_a_regressed_ceiling_fails_without_overwriting_the_file(
    operation: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve stricter recorded limits when actual transport falls short."""
    tool = _tool()
    destination = tmp_path / "ceiling.toml"
    original = b"[lanes.rust-batch]\nmax_dropped = -1\nmin_complete = 10000\n"
    destination.write_bytes(original)

    assert tool.main([operation, "--ceiling", str(destination)]) == 1
    assert "transport got worse" in capsys.readouterr().err
    assert destination.read_bytes() == original


def test_an_existing_ceiling_regenerates_without_changing_its_values(tmp_path: Path) -> None:
    """Emit descriptive headers while retaining the measured transport limits."""
    tool = _tool()
    destination = tmp_path / "ceiling.toml"
    original = (REPO_ROOT / "tools" / "runtime_state_ceiling.toml").read_bytes()
    destination.write_bytes(original)

    assert tool.main(["--update-ceiling", "--ceiling", str(destination)]) == 0
    generated = destination.read_bytes()
    assert generated == original
    assert tool.tomllib.loads(generated.decode()) == tool.tomllib.loads(original.decode())

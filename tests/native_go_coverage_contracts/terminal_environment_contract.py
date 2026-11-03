# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual inherited ABI child environment custody contracts

"""Conserve actual Readline metadata and refuse real native or recorded terminal drift."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.go_coverage_environment import (
    declare_terminal_environment,
    environment_mapping,
    terminal_environment,
)
from tools.go_mat_coverage_build import NativeGo


def test_actual_terminal_metadata_is_declared_in_every_real_consumer_base(
    native_job: Path,
) -> None:
    """Read actual Readline C values and conserve them across all real phase declarations."""
    actual = terminal_environment()
    assert set(actual) == {"COLUMNS", "LINES"}
    initialization = json.loads((native_job / "readline-initialization.json").read_text())
    assert initialization["terminal_environment"] == actual
    assert initialization["module_loaded_before"] is False
    assert (
        hashlib.sha256(Path(initialization["module_path"]).read_bytes()).hexdigest()
        == initialization["module_sha256"]
    )
    assert initialization["producer_pid"] > 0
    assert any(
        frame["function"] == "python_environment"
        for frame in initialization["actual_declaration_stack"]
    )
    for path in (native_job / "environments").glob("*.json"):
        mapping = environment_mapping(json.loads(path.read_text()))
        assert {key: mapping[key] for key in actual} == actual
    for path in (native_job / "process-counters").glob("*/*/observations.jsonl"):
        for row in [json.loads(line) for line in path.read_text().splitlines()]:
            if row["event"] == "closed_environment":
                assert row["terminal_environment"] == actual
                assert set(actual).issubset(row["keys"])


@pytest.mark.parametrize("key", ["COLUMNS", "LINES"])
@pytest.mark.parametrize("damage", [None, "", "0", "-1", "01", "1.5", "invalid", "\u0661"])
def test_actual_c_terminal_metadata_refuses_missing_or_invalid_values(
    key: str, damage: str | None
) -> None:
    """Mutate the real libc field, require refusal, and restore its actual original value."""
    original = terminal_environment()
    try:
        if damage is None:
            os.unsetenv(key)
        else:
            os.putenv(key, damage)
        with pytest.raises(RuntimeError, match="Readline terminal field"):
            terminal_environment()
    finally:
        os.putenv(key, original[key])
    assert terminal_environment() == original


def test_actual_native_terminal_drift_refuses_a_second_registration(
    native_job: Path, native_builder: NativeGo
) -> None:
    """A real libc-only metadata change cannot pass a later registration with stale Python cache."""
    environment = environment_mapping(
        json.loads((native_job / "environments/contracts.json").read_text())
    )
    driver = native_builder.job / "artifacts/terminal-drift-driver.py"
    driver.write_text(
        "import os,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_environment import terminal_environment;"
        "from tools.go_coverage_images import register;"
        "job=Path(sys.argv[2]);register(job,'contracts');"
        "original=terminal_environment();"
        "assert all(os.environ[k]==v for k,v in original.items());"
        "os.putenv('COLUMNS',str(int(original['COLUMNS'])+1))\n"
        "try:\n"
        "    try:\n"
        "        register(job,'contracts')\n"
        "    except RuntimeError as error:\n"
        "        assert 'Readline terminal metadata changed' in str(error)\n"
        "    else:\n"
        "        raise AssertionError('Actual native terminal drift was accepted')\n"
        "finally:\n"
        "    os.putenv('COLUMNS',original['COLUMNS'])\n"
    )
    assert (
        native_builder.command(
            "terminal-drift-refusal",
            native_builder.python_script(driver, [str(native_builder.repository), str(native_job)]),
            native_builder.job,
            environment=environment,
        )
        == 0
    )


@pytest.mark.parametrize("field", ["terminal_environment", "module_path", "module_sha256"])
def test_actual_first_terminal_declaration_cannot_be_replaced(tmp_path: Path, field: str) -> None:
    """Damaged copies of real first-initialization metadata cannot authorize a later phase."""
    actual = declare_terminal_environment(tmp_path)
    assert declare_terminal_environment(tmp_path) == actual
    path = tmp_path / "readline-initialization.json"
    changed = json.loads(path.read_text())
    changed.pop(field)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="initialization changed"):
        declare_terminal_environment(tmp_path)

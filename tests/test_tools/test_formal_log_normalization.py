# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External formal receipt consistency checks

"""Check reversible identity normalization on an actual native proof log."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
import re
import pwd
import subprocess
import sys

import pytest

from tests.formal_native_evidence_support import NativeEvidence, collect_native
from tools.formal_evidence_validation import object_fields, read_object
from tools.formal_log_normalization import NORMALIZER_ID, PLACEHOLDERS, main, normalize_native_log


@pytest.fixture(scope="module")
def native_log(tmp_path_factory: pytest.TempPathFactory) -> NativeEvidence:
    """Collect an actual finite SBY proof; preserve its raw log unchanged."""
    return collect_native(tmp_path_factory.mktemp("normalization-native"))


def _identities(native: NativeEvidence) -> dict[str, str]:
    """Identify real workspace, host and user plus the native test's unit name."""
    measurement = object_fields(read_object(native.repository / "receipt.json")["measurement"])
    return {
        "workspace": str(native.repository),
        "unit": str(measurement["unit"]),
        "hostname": socket.gethostname(),
        "username": pwd.getpwuid(os.getuid()).pw_name,
    }


def test_real_native_log_golden_byte_ranges(native_log: NativeEvidence) -> None:
    """Keep every byte outside nominated identity ranges, including proof timing."""
    identities = _identities(native_log)
    raw_path = native_log.repository / "native.log"
    raw = raw_path.read_bytes()
    normalized = normalize_native_log(raw, identities)
    restored = normalized
    for name, token in reversed(list(PLACEHOLDERS.items())):
        restored = restored.replace(token, identities[name].encode())
    assert restored == raw
    assert hashlib.sha256(restored).digest() == hashlib.sha256(raw).digest()
    assert len(normalized.splitlines(keepends=True)) == len(raw.splitlines(keepends=True))
    original_ranges = re.compile(
        b"|".join(re.escape(value.encode()) for value in identities.values())
    )
    substituted_ranges = re.compile(b"|".join(re.escape(token) for token in PLACEHOLDERS.values()))
    for original, public in zip(
        raw.splitlines(keepends=True), normalized.splitlines(keepends=True), strict=True
    ):
        assert original_ranges.sub(b"", original) == substituted_ranges.sub(b"", public)
    assert b"DONE (PASS, rc=0)" in normalized
    assert raw_path.read_bytes() == raw


@pytest.mark.parametrize(
    "name,value",
    [
        ("workspace", "relative/path"),
        ("workspace", "/"),
        ("workspace", "/private/path with space"),
        ("unit", "other.service"),
        ("hostname", ""),
        ("username", "INVALID USER"),
    ],
)
def test_identity_refusals(native_log: NativeEvidence, name: str, value: str) -> None:
    """Refuse identity ranges that do not match their concrete field types."""
    identities = _identities(native_log)
    identities[name] = value
    with pytest.raises(ValueError):
        normalize_native_log((native_log.repository / "native.log").read_bytes(), identities)


def test_placeholder_and_inverse_collisions_refuse(native_log: NativeEvidence) -> None:
    """Reject ambiguous tokens and substitutions that would alter other ranges."""
    identities = _identities(native_log)
    raw = (native_log.repository / "native.log").read_bytes()
    with pytest.raises(ValueError, match="Incomplete"):
        normalize_native_log(raw, {})
    with pytest.raises(ValueError, match="already contains"):
        normalize_native_log(raw + b"<FORMAL_HOST>", identities)
    identities["hostname"] = "FORMAL"
    with pytest.raises(ValueError, match="collides with fixed"):
        normalize_native_log(raw, identities)


def test_real_cli_distinct_digests_and_raw_custody(
    native_log: NativeEvidence, tmp_path: Path
) -> None:
    """Exercise the module CLI on real raw bytes and refuse overwrite attempts."""
    identities = _identities(native_log)
    raw = native_log.repository / "native.log"
    output = tmp_path / "public.log"
    args = [str(raw), str(output)]
    for name, value in identities.items():
        args.extend(["--" + name, value])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "tools.formal_log_normalization",
            *args,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    custody = json.loads(result.stdout)
    assert custody == {
        "normalizer_id": NORMALIZER_ID,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "normalized_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    assert custody["raw_sha256"] != custody["normalized_sha256"]
    assert main(args) == 2
    args[1] = str(raw)
    assert main(args) == 2

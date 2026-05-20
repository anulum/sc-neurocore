# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_GO_DIRECTIVE = (1, 26, 3)


def _parse_go_directive(path: Path) -> tuple[int, int, int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("go "):
            continue
        version = stripped.split(maxsplit=1)[1]
        parts = [int(part) for part in version.split(".")]
        if len(parts) == 2:
            parts.append(0)
        return tuple(parts[:3])  # type: ignore[return-value]
    raise AssertionError(f"missing go directive in {path}")


def test_go_modules_use_current_security_baseline() -> None:
    go_mods = sorted((REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go").rglob("go.mod"))
    assert go_mods

    stale = [
        str(path.relative_to(REPO_ROOT))
        for path in go_mods
        if _parse_go_directive(path) < MINIMUM_GO_DIRECTIVE
    ]

    assert stale == []


def test_aer_router_module_uses_single_non_test_package() -> None:
    aer_router = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go" / "services" / "aer_router"
    packages = set()
    for source in sorted(aer_router.glob("*.go")):
        if source.name.endswith("_test.go"):
            continue
        match = re.search(r"^package\s+(\w+)$", source.read_text(encoding="utf-8"), re.MULTILINE)
        assert match is not None, source
        packages.add(match.group(1))

    assert packages == {"main"}

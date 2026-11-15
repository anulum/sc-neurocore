# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Keep the Rust safety crate's module list, ``safety/lib.rs``, aligned with checked-in sources."""

from __future__ import annotations

from pathlib import Path
import re

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFETY_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "rust" / "safety"
_MODULE_PATTERN = re.compile(r"^pub mod ([A-Za-z0-9_]+);$")


def _declared_modules(path: Path) -> tuple[str, ...]:
    """Return ordinary public module declarations in source order."""
    modules: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _MODULE_PATTERN.fullmatch(line.strip())
        if match:
            modules.append(match.group(1))
    return tuple(modules)


def test_safety_module_registry_references_real_sources() -> None:
    """Every declaration of the crate's one module list, ``lib.rs``, must resolve."""
    modules = _declared_modules(_SAFETY_ROOT / "lib.rs")
    missing = [
        module
        for module in modules
        if not (_SAFETY_ROOT / f"{module}.rs").is_file()
        and not (_SAFETY_ROOT / module / "mod.rs").is_file()
    ]

    assert modules
    assert missing == []

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SPDX header audit tests

"""Tests for the source/config SPDX header audit tool."""

from __future__ import annotations

from pathlib import Path

from tools import spdx_header_audit


def _write(root: Path, relative: str, text: str) -> None:
    """Write a test file below ``root``."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_classifies_comment_safe_source_as_direct_header(tmp_path: Path) -> None:
    """Comment-safe source files require a direct seven-line header."""

    path = "studio/frontend/src/App.tsx"
    _write(tmp_path, path, "export default function App() { return null; }\n")

    record = spdx_header_audit.classify_path(tmp_path, path)

    assert record.kind is spdx_header_audit.CoverageKind.DIRECT_HEADER
    assert record.missing_direct_header


def test_classifies_json_and_generated_artifacts_without_rewriting(tmp_path: Path) -> None:
    """JSON data and generated evidence are separated from direct headers."""

    schema_path = "src/sc_neurocore/neurons/model_schemas/lif.json"
    generated_path = "benchmarks/results/bench_lif.json"
    _write(tmp_path, schema_path, '{"type": "object"}\n')
    _write(tmp_path, generated_path, '{"elapsed": 1.0}\n')

    schema_record = spdx_header_audit.classify_path(tmp_path, schema_path)
    generated_record = spdx_header_audit.classify_path(tmp_path, generated_path)

    assert schema_record.kind is spdx_header_audit.CoverageKind.REUSE_METADATA
    assert generated_record.kind is spdx_header_audit.CoverageKind.GENERATED_OR_DATA
    assert not schema_record.missing_direct_header
    assert not generated_record.missing_direct_header


def test_apply_direct_header_preserves_shebang(tmp_path: Path) -> None:
    """Shell scripts keep the interpreter line before the SPDX header."""

    path = "scripts/run_lane.sh"
    _write(tmp_path, path, "#!/usr/bin/env bash\nset -euo pipefail\n")

    assert spdx_header_audit.apply_direct_header(tmp_path, path)

    lines = (tmp_path / path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "SPDX-License-Identifier: AGPL-3.0-or-later" in lines[1]
    assert "ORCID: 0009-0009-3560-0851" in "\n".join(lines[:8])


def test_apply_direct_header_preserves_typescript_reference(tmp_path: Path) -> None:
    """TypeScript reference directives remain before the inserted header."""

    path = "studio/frontend/src/vite-env.d.ts"
    _write(tmp_path, path, '/// <reference types="vite/client" />\n')

    assert spdx_header_audit.apply_direct_header(tmp_path, path)

    lines = (tmp_path / path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == '/// <reference types="vite/client" />'
    assert "SPDX-License-Identifier: AGPL-3.0-or-later" in lines[1]


def test_missing_direct_header_paths_ignores_reuse_and_generated_files(tmp_path: Path) -> None:
    """The missing list contains only direct-header policy targets."""

    _write(tmp_path, "src/demo.go", "package demo\n")
    _write(tmp_path, "src/schema.json", "{}\n")
    _write(tmp_path, "sc_shd_pynq/generated.v", "module generated; endmodule\n")

    missing = spdx_header_audit.missing_direct_header_paths(
        tmp_path,
        paths=("src/demo.go", "src/schema.json", "sc_shd_pynq/generated.v"),
    )

    assert missing == ["src/demo.go"]

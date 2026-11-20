# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for the accelerated-lane stub triage

"""Hold the stub triage to its rule: markers, thresholds, classes, readings and the committed record."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import accel_stub_triage
from tools.accel_stub_triage import (
    DEFAULT_OUTPUT,
    OPERATORS,
    PURE_OPERATORS,
    build,
    classify,
    main,
    markers,
    render,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _operators(count: int) -> str:
    """Return a function body line with ``count`` multiplications."""
    return "    return " + " * ".join(["x"] * (count + 1)) + "\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("import numpy as np\n\ndef f(x):\n    return np.exp(x)\n", "compute"),
        ("import scipy.signal\n\ndef f(x):\n    return scipy.signal.lfilter(x)\n", "compute"),
        ("from fastapi import FastAPI\n\napp = FastAPI()\n", "application"),
        ("import subprocess\n\ndef run():\n    return subprocess.run(['x'])\n", "application"),
        (
            "import argparse\n\ndef main():\n    return argparse.ArgumentParser()\n\n"
            "if __name__ == '__main__':\n    main()\n",
            "application",
        ),
        ("import argparse\n\ndef parser():\n    return argparse.ArgumentParser()\n", "unclear"),
        (
            "import numpy as np\nimport matplotlib.pyplot as plt\n\ndef f(x):\n    plt.plot(np.sin(x))\n",
            "unclear",
        ),
        ("def f(x):\n    return x + 1\n", "unclear"),
        ("import numpy as np\n\nZEROS = np.zeros(3)\n", "unclear"),
    ],
    ids=[
        "numpy-call-in-a-function",
        "scipy-attribute-chain",
        "web-framework",
        "process-client",
        "argparse-with-a-main-block",
        "argparse-without-a-main-block",
        "both-markers",
        "neither-marker",
        "numpy-only-at-module-level",
    ],
)
def test_each_rule_classes_its_example(text: str, expected: str) -> None:
    """Each class is reached by its own source, and both or neither markers stay unclear."""
    assert classify(markers(text)) == expected


def test_the_operator_thresholds_are_exact() -> None:
    """A pure numeric call needs PURE_OPERATORS operators; operators alone need OPERATORS."""
    pure = "import math\n\ndef f(x):\n    math.sqrt(x)\n"
    assert classify(markers(pure + _operators(PURE_OPERATORS - 1))) == "unclear"
    assert classify(markers(pure + _operators(PURE_OPERATORS))) == "compute"
    alone = "def f(x):\n"
    assert classify(markers(alone + _operators(OPERATORS - 1))) == "unclear"
    assert classify(markers(alone + _operators(OPERATORS))) == "compute"


def test_operators_in_a_nested_function_count_once() -> None:
    """An operator inside a nested function is counted once, not once per enclosing function."""
    text = "def outer(x):\n    def inner(y):\n        return y * y\n    return inner(x)\n"
    assert markers(text)["operators"] == 1


def test_a_threshold_never_makes_application() -> None:
    """Below every threshold an app import alone is application only without numeric work."""
    text = "import socket\nimport numpy as np\n\ndef f(x):\n    return np.abs(x)\n"
    assert classify(markers(text)) == "unclear"


def test_the_committed_triage_is_current() -> None:
    """The committed record equals a fresh build of the repository."""
    committed = (REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
    assert committed == render(build(REPO_ROOT))


def test_the_record_holds_together() -> None:
    """Classes partition the originals; the deletion list is exactly the application stubs; the order is compute."""
    record = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    counts = record["counts"]
    assert (
        sum(counts[name] for name in ("compute", "application", "unclear")) == counts["originals"]
    )
    application = [o for o in record["originals"] if o["class"] == "application"]
    assert [entry["path"] for entry in record["deletion_list"]] == [
        stub["path"] for original in application for stub in original["stubs"]
    ]
    compute = {o["module"] for o in record["originals"] if o["class"] == "compute"}
    assert set(record["compute_order"]) == compute
    used = [o["used"] for o in record["originals"] if o["class"] == "compute"]
    order = {module: index for index, module in enumerate(record["compute_order"])}
    by_module = {o["module"]: o for o in record["originals"]}
    flags = [by_module[module]["used"] for module in sorted(order, key=order.__getitem__)]
    assert flags == sorted(used, reverse=True)


def test_a_reading_of_a_module_the_rule_decides_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reading cannot override the rule silently."""
    record = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    decided = next(o["module"] for o in record["originals"] if o["class"] == "compute")
    monkeypatch.setitem(
        accel_stub_triage.READINGS,
        decided,
        {"class": "application", "reason": "a test", "reader": "test"},
    )
    with pytest.raises(ValueError, match="overrides the rule"):
        build(REPO_ROOT)


def test_a_reading_of_a_module_no_stub_copies_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reading must name an original a stub copies."""
    monkeypatch.setitem(
        accel_stub_triage.READINGS,
        "src/sc_neurocore/not_copied_by_any_stub.py",
        {"class": "compute", "reason": "a test", "reader": "test"},
    )
    with pytest.raises(ValueError, match="no stub copies"):
        build(REPO_ROOT)


def test_an_unresolved_ambiguous_stub_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ambiguous stub without a valid recorded resolution stops the triage."""
    stub = next(iter(accel_stub_triage.AMBIGUOUS_STUBS))
    monkeypatch.delitem(accel_stub_triage.AMBIGUOUS_STUBS, stub)
    with pytest.raises(ValueError, match="no valid recorded resolution"):
        build(REPO_ROOT)


def test_the_command_writes_checks_and_summarises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--write`` then ``--check`` pass; a changed file drifts; a refused tree exits 2."""
    output = tmp_path / "triage.json"
    assert main(["--write", "--output", str(output)]) == 0
    assert main(["--check", "--summary", "--output", str(output)]) == 0
    assert "originals:" in capsys.readouterr().out
    output.write_text("{}\n", encoding="utf-8")
    assert main(["--check", "--output", str(output)]) == 1
    output.unlink()
    assert main(["--check", "--output", str(output)]) == 1

    def refused(root: Path) -> accel_stub_triage.Inventory:
        """Refuse the tree as a stale reading would."""
        raise ValueError("refused")

    assert main(["--check", "--output", str(output)], builder=refused) == 2


def test_stubs_without_a_module_are_listed_and_a_module_is_read_once(tmp_path: Path) -> None:
    """A Mojo stub with no module and a headerless Julia file are unmapped; two Julia files share one read."""
    module = tmp_path / "src" / "sc_neurocore" / "kernel.py"
    julia = tmp_path / "src" / "sc_neurocore" / "accel" / "julia"
    julia.mkdir(parents=True)
    module.write_text("import numpy as np\n\ndef f(x):\n    return np.exp(x)\n", encoding="utf-8")
    header = "# SC-NeuroCore — Julia acceleration for kernel\nfunction f(x) x end\n"
    (julia / "a.jl").write_text(header, encoding="utf-8")
    (julia / "b.jl").write_text(header, encoding="utf-8")
    (julia / "c.jl").write_text("function g(x) x end\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("PATH = 'a.jl'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    flags = {
        "parses": True,
        "loaded": False,
        "authoritative": False,
        "mirror": False,
        "loaders": [],
    }
    lane_julia: dict[str, Any] = {
        "sources": [
            {"path": f"src/sc_neurocore/accel/julia/{name}.jl", **flags} for name in ("a", "b", "c")
        ]
    }
    lane_mojo: dict[str, Any] = {
        "sources": [
            {
                "path": "src/sc_neurocore/accel/mojo/kernels/lost.mojo",
                "class": "translated_stub",
                "transcribed_module": None,
                "transcription_rule": "none",
                "transcription_candidates": [],
                "verbatim_line_share": None,
                "loaders": [],
                "declared": "mirror",
            }
        ]
    }
    grouped, unmapped = accel_stub_triage.lane_stubs(tmp_path, lane_mojo, lane_julia)
    assert unmapped == [
        "src/sc_neurocore/accel/julia/c.jl",
        "src/sc_neurocore/accel/mojo/kernels/lost.mojo",
    ]
    stubs = grouped["src/sc_neurocore/kernel.py"]
    assert [(stub["lane"], stub["path"]) for stub in stubs] == [
        ("julia_parseable_unloaded", "src/sc_neurocore/accel/julia/a.jl"),
        ("julia_parseable_unloaded", "src/sc_neurocore/accel/julia/b.jl"),
    ]
    assert stubs[0]["named_by"] == ["tests/test_a.py"]


def test_the_basis_names_the_first_numeric_marker_and_the_variant_is_recorded() -> None:
    """Library calls are basis a, pure calls b, operators alone c; the a-only variant keeps every application."""
    assert markers("import numpy as np\n\ndef f(x):\n    return np.exp(x)\n")["basis"] == "a"
    pure = "import math\n\ndef f(x):\n    math.sqrt(x)\n"
    assert markers(pure + _operators(PURE_OPERATORS))["basis"] == "b"
    assert markers("def f(x):\n" + _operators(OPERATORS))["basis"] == "c"
    assert markers("def f(x):\n    return x\n")["basis"] is None
    record = json.loads((REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    variant = record["variant_a_only"]
    assert (
        sum(variant[name] for name in ("compute", "application", "unclear"))
        == record["counts"]["originals"]
    )
    full = {
        o["module"]
        for o in record["originals"]
        if o["class"] == "application" and o["decided_by"] == "rule"
    }
    assert full <= set(variant["application_modules"])
    ranks = [
        (not by["used"], "abc".index(by["markers"]["basis"] or "c"))
        for by in (
            next(o for o in record["originals"] if o["module"] == module)
            for module in record["compute_order"]
        )
    ]
    assert ranks == sorted(ranks)


def test_a_name_imported_from_a_numeric_library_marks_numeric_work() -> None:
    """``from numpy import exp`` then ``exp(x)`` is a numpy call, basis a."""
    found = markers("from numpy import exp\n\ndef f(x):\n    return exp(x)\n")
    assert (found["basis"], classify(found)) == ("a", "compute")


def test_a_stub_is_named_only_at_a_path_boundary() -> None:
    """``lif.jl`` is not found inside ``adaptive_lif.jl``, but is found after a separator or a quote."""
    texts = {
        "tests/a.py": "PATH = 'neurons/adaptive_lif.jl'\n",
        "tests/b.py": "PATH = 'neurons/lif.jl'\n",
        "tests/c.py": "PATH = 'lif.jl'\n",
    }
    root = "src/sc_neurocore/accel/julia"
    tokens = accel_stub_triage._path_tokens(texts)
    assert accel_stub_triage._named_by(f"{root}/neurons/lif.jl", root, tokens) == ["tests/b.py"]
    assert accel_stub_triage._named_by(f"{root}/lif.jl", root, tokens) == [
        "tests/b.py",
        "tests/c.py",
    ]


def test_an_unclear_original_stays_out_of_the_deletion_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without its reading, a module the rule leaves unclear is unclear, and none of its stubs is deleted."""
    module = next(
        name
        for name, reading in accel_stub_triage.READINGS.items()
        if reading["class"] == "application"
    )
    monkeypatch.delitem(accel_stub_triage.READINGS, module)
    triage = build(REPO_ROOT)
    original = next(o for o in triage["originals"] if o["module"] == module)
    assert (original["class"], original["decided_by"]) == ("unclear", "rule")
    assert module not in {entry["module"] for entry in triage["deletion_list"]}

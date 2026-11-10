# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Every shipped subpackage imports without the source-only packages

"""Keep every subpackage the wheel ships importable without the packages it leaves out.

``pyproject.toml`` excludes some research packages from the wheel. A shipped
module that imported one of them at module level made its whole package fail
to import from an installed wheel; ``sc_neurocore.identity`` and
``sc_neurocore.experimental`` did. A subprocess blocks every excluded package,
as an installed wheel lacks them, and imports every shipped top-level
subpackage. Only failures caused by a blocked package count here; a missing
optional third-party dependency is a different, declared boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sc_neurocore._source_checkout import require_source_checkout_package

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = r"""
import importlib, importlib.abc, json, pkgutil, sys
from pathlib import Path
from types import SimpleNamespace

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

root = Path(sys.argv[1])
find = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"]["find"]
blocked = sorted({entry.removesuffix(".*") for entry in find["exclude"]})


def _is_blocked(name):
    return any(name == b or name.startswith(b + ".") for b in blocked)


class _WheelLacks(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _is_blocked(name):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _WheelLacks())
import sc_neurocore

shipped = sorted(
    m.name for m in pkgutil.iter_modules(sc_neurocore.__path__)
    if m.ispkg and not _is_blocked(f"sc_neurocore.{m.name}")
)
failures = {}
for name in shipped:
    try:
        importlib.import_module(f"sc_neurocore.{name}")
    except ModuleNotFoundError as exc:
        if exc.name and _is_blocked(exc.name):
            failures[name] = str(exc)
    except Exception:
        pass

try:
    importlib.import_module("sc_neurocore.analysis")
    blocker_works = False
except ModuleNotFoundError:
    blocker_works = True

refusals = {}
try:
    from sc_neurocore.experimental import make_heat_cosine_mode_route
    make_heat_cosine_mode_route().baseline(0.3, 0.01)
    refusals["heat"] = None
except ModuleNotFoundError as exc:
    refusals["heat"] = str(exc)
try:
    from sc_neurocore.identity import StateDecoder
    history = [[1, 0, 1, 0]] * 12
    decoder = StateDecoder(SimpleNamespace(spike_history=history, n_cortical=4))
    decoder.extract_connectivity_signature()
    refusals["decoder"] = None
except ModuleNotFoundError as exc:
    refusals["decoder"] = str(exc)

print(json.dumps({
    "blocked": blocked,
    "shipped": shipped,
    "failures": failures,
    "blocker_works": blocker_works,
    "refusals": refusals,
}))
"""


@pytest.fixture(scope="module")
def probe() -> dict[str, object]:
    """Run the import probe once against this checkout's source tree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result: dict[str, object] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


def test_every_shipped_subpackage_imports_without_the_source_only_packages(
    probe: dict[str, object],
) -> None:
    """No shipped top-level subpackage needs an excluded package to be imported."""
    blocked = probe["blocked"]
    shipped = probe["shipped"]
    assert isinstance(blocked, list) and isinstance(shipped, list)
    assert {"sc_neurocore.analysis", "sc_neurocore.physics"} <= set(blocked)
    assert {"identity", "experimental"} <= set(shipped)
    assert probe["failures"] == {}


def test_the_probe_really_blocks_the_source_only_packages(probe: dict[str, object]) -> None:
    """Negative control: inside the probe an excluded package cannot be imported."""
    assert probe["blocker_works"] is True


def test_features_that_need_a_source_only_package_refuse_with_the_reason(
    probe: dict[str, object],
) -> None:
    """The heat route and the decoder statistics name the missing package and why."""
    refusals = probe["refusals"]
    assert isinstance(refusals, dict)
    heat, decoder = refusals["heat"], refusals["decoder"]
    assert isinstance(heat, str) and "sc_neurocore.physics.heat" in heat
    assert isinstance(decoder, str) and "StateDecoder.extract_connectivity_signature" in decoder
    for message in (heat, decoder):
        assert "only in a source checkout" in message


def test_a_present_package_passes() -> None:
    """A package that is installed is imported and no refusal is raised."""
    require_source_checkout_package("sc_neurocore.utils", "Any feature")


def test_an_absent_package_is_refused_with_the_feature_named() -> None:
    """An absent package gives a refusal that names the feature and the package."""
    package = "sc_neurocore_absent_package_for_this_test"
    with pytest.raises(ModuleNotFoundError) as caught:
        require_source_checkout_package(package, "The feature under test")
    assert caught.value.name == package
    assert str(caught.value) == (
        f"The feature under test needs {package}, which ships only in a source checkout "
        "of SC-NeuroCore, not in the wheel."
    )


def test_a_missing_dependency_inside_a_present_package_is_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third-party module missing inside the package is raised unchanged."""
    package = tmp_path / "source_checkout_probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import third_party_module_absent_for_this_test\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "source_checkout_probe_pkg", raising=False)
    with pytest.raises(ModuleNotFoundError) as caught:
        require_source_checkout_package("source_checkout_probe_pkg", "The feature under test")
    assert caught.value.name == "third_party_module_absent_for_this_test"
    assert "source checkout" not in str(caught.value)

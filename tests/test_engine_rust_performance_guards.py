# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust performance test guard contracts

"""Static guards for the engine's timed Rust unit tests.

A timed test is held to a budget derived from the host that runs it (``engine/src/neurons/host_budget.rs``), with
reference ratios kept in ``engine/test_host_profiles.toml``. These tests hold the profile file to the rule that
produced it and every budgeted test to an entry in it. The literal millisecond limits not yet converted are listed,
so a new one is refused.
"""

from __future__ import annotations

import re
import statistics
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCES = REPO_ROOT / "engine" / "src"
PROFILES = REPO_ROOT / "engine" / "test_host_profiles.toml"
POLICY_MARGIN = 5.0
BUILDS = ("debug", "release")
MINIMUM_RUNS = 5
HOST_FIELDS = {"cpu", "logical_cpus", "kernel", "rustc", "measured_at", "load_average_before"}

_CALL = re.compile(r'host_budget::assert_within\(\s*module_path!\(\),\s*"(?P<name>\w+)"')
_FUNCTION = re.compile(r"\bfn (?P<name>\w+)\(")
_LITERAL_DEADLINE = re.compile(r"as_millis\(\)\s*<")
_PATH_MODULE = re.compile(
    r'#\[path\s*=\s*"(?P<file>[^"]+)"\]\s*(?:pub(?:\([^)]*\))?\s+)?mod (?P<module>\w+);'
)

# Literal millisecond limits in modules whose documentation units were still open when the first modules were
# converted. Each later unit converts a module's tests and removes its line here.
_LITERAL_DEADLINES_LEFT = {
    "engine/src/neuron/adex/tests.rs": 1,
    "engine/src/neuron/exp_if_tests.rs": 1,
    "engine/src/neurons/ermentrout_kopell_map.rs": 1,
    "engine/src/neurons/population/brunel_network_tests.rs": 1,
    "engine/src/neurons/population/montbrio_mean_field_tests.rs": 1,
    "engine/src/neurons/population/tum_network_tests.rs": 1,
}


def _profiles() -> dict[str, object]:
    """Read ``engine/test_host_profiles.toml``."""
    return tomllib.loads(PROFILES.read_text(encoding="utf-8"))


def _module_path(source: Path) -> str:
    """Return the Rust module path of an engine source file by its place, without the crate name.

    The crate root ``lib.rs`` has the empty path.
    """
    parts = list(source.relative_to(ENGINE_SOURCES).with_suffix("").parts)
    if parts[-1] == "mod" or parts == ["lib"]:
        parts.pop()
    return "::".join(parts)


def _join(*paths: str) -> str:
    """Join module paths with ``::``, leaving out the crate root's empty path."""
    return "::".join(path for path in paths if path)


def _path_modules() -> dict[Path, str]:
    """Map each file a ``#[path = "..."] mod <name>;`` declaration includes to its module path.

    The path is relative to the declaring file's folder, as Rust resolves it for a module declared outside any inline
    module block. A declaration may carry a visibility. A declaring file that is itself included by ``#[path]`` lends
    its own included path, so a chain of such files resolves back to a file named by its place. Unit 12 and later
    units moved some files' tests to ``<name>_tests.rs`` this way, and the bindings include model files this way.
    """
    declarations = {}
    for source in sorted(ENGINE_SOURCES.rglob("*.rs")):
        for declaration in _PATH_MODULE.finditer(source.read_text(encoding="utf-8")):
            target = (source.parent / declaration["file"]).resolve()
            declarations[target] = (source, declaration["module"])

    def resolve(source: Path) -> str:
        declared = declarations.get(source.resolve())
        if declared is None:
            return _module_path(source)
        parent, module = declared
        return _join(resolve(parent), module)

    return {target: resolve(target) for target in declarations}


def _budgeted_tests() -> list[tuple[str, str, str, str]]:
    """Find every budgeted test as (source, key, enclosing function, attributes above that function).

    A budget sits in a file that is itself a module called ``tests`` (declared ``mod tests;``, with or without
    ``#[path]``), or after an inline ``mod tests`` in any other file; its key is the test module's path, as
    ``module_path!()`` gives it, and the test's name.
    """
    found = []
    included = _path_modules()
    for source in sorted(ENGINE_SOURCES.rglob("*.rs")):
        text = source.read_text(encoding="utf-8")
        module = included.get(source.resolve(), _module_path(source))
        tests_file = module.rsplit("::", 1)[-1] == "tests"
        tests_module = text.find("mod tests {")
        for call in _CALL.finditer(text):
            if tests_file:
                tests_path = module
            else:
                assert tests_module != -1 and tests_module < call.start(), (
                    f"{source}: a budget outside `mod tests`"
                )
                tests_path = _join(module, "tests")
            functions = list(_FUNCTION.finditer(text, 0, call.start()))
            enclosing = functions[-1]
            line_start = text.rfind("\n", 0, enclosing.start())
            previous_end = text.rfind("}", 0, line_start)
            attributes = text[previous_end + 1 : line_start]
            key = f"{tests_path}::{call['name']}"
            found.append((str(source.relative_to(REPO_ROOT)), key, enclosing["name"], attributes))
    return found


def _round_up(value: Decimal) -> Decimal:
    """Round a positive number up to two significant figures."""
    step = Decimal(1).scaleb(value.adjusted() - 1)
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def test_round_up_keeps_two_significant_figures_and_never_rounds_down() -> None:
    """The rounding the ratios are set by."""
    assert _round_up(Decimal("0.5234")) == Decimal("0.53")
    assert _round_up(Decimal("0.53")) == Decimal("0.53")
    assert _round_up(Decimal("12.01")) == Decimal("13")
    assert _round_up(Decimal("0.0123")) == Decimal("0.013")
    assert _round_up(Decimal("5")) == Decimal("5.0")


def test_the_budgeted_tests_are_the_converted_ones() -> None:
    """The modules converted so far hold 34 budgeted tests, the five motor tests among them; none is dropped."""
    assert len(_budgeted_tests()) == 34


def test_every_budget_is_named_after_its_test_function() -> None:
    """A budget's name is its enclosing test function, so the profile key is the name ``cargo test`` prints."""
    for source, key, function, _ in _budgeted_tests():
        assert key.endswith(f"::{function}"), f"{source}: budget {key} sits in fn {function}"


def test_budgeted_tests_run_by_default() -> None:
    """A budgeted test is not opt-in: the budget, not an ignore marker, keeps it from failing on a busy host."""
    for source, key, _, attributes in _budgeted_tests():
        assert "#[test]" in attributes, f"{source}: {key} is not a test"
        assert "#[ignore" not in attributes, f"{source}: {key} is ignored"


def test_every_budgeted_test_has_a_ratio_and_every_ratio_a_test() -> None:
    """The profile's ratios and the budgeted tests are the same set."""
    keys = sorted(key for _, key, _, _ in _budgeted_tests())
    ratios = _profiles()["ratio"]
    assert isinstance(ratios, dict)
    assert sorted(ratios) == keys
    assert len(set(keys)) == len(keys)


def test_each_ratio_is_the_largest_reference_median_rounded_up() -> None:
    """Every test is measured on every reference host in both builds, and its ratio follows from those runs."""
    profiles = _profiles()
    hosts = profiles["reference_hosts"]
    measurements = profiles["measurement"]
    ratios = profiles["ratio"]
    assert isinstance(hosts, dict) and isinstance(measurements, dict) and isinstance(ratios, dict)
    assert sorted(measurements) == sorted(hosts)
    for host, builds in measurements.items():
        assert sorted(builds) == sorted(BUILDS), host
        for build, runs in builds.items():
            assert sorted(runs) == sorted(ratios), f"{host} {build}"
            for key, values in runs.items():
                assert len(values) >= MINIMUM_RUNS, f"{host} {build} {key}"
                assert all(value > 0 for value in values), f"{host} {build} {key}"
    for key, ratio in ratios.items():
        for build in BUILDS:
            medians = [
                Decimal(repr(statistics.median(measurements[host][build][key]))) for host in hosts
            ]
            expected = _round_up(max(medians))
            assert Decimal(repr(ratio[build])) == expected, (
                f"{key} {build}: {ratio[build]} != {expected}"
            )


def test_the_margin_is_the_policy_value_and_the_file_holds_nothing_else() -> None:
    """One policy margin; the reader in the engine reads only the margin and the ratios."""
    profiles = _profiles()
    assert profiles["margin"] == POLICY_MARGIN
    assert profiles["schema_version"] == 1
    assert sorted(profiles) == [
        "margin",
        "measurement",
        "ratio",
        "reference_hosts",
        "schema_version",
    ]
    ratios = profiles["ratio"]
    assert isinstance(ratios, dict)
    for key, ratio in ratios.items():
        assert sorted(ratio) == sorted(BUILDS), key


def test_every_reference_host_records_its_profile() -> None:
    """A reference host is described well enough to tell which host and toolchain the ratios came from."""
    hosts = _profiles()["reference_hosts"]
    assert isinstance(hosts, dict) and hosts
    for name, host in hosts.items():
        assert set(host) == HOST_FIELDS, name
        assert isinstance(host["logical_cpus"], int) and host["logical_cpus"] > 0, name
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", host["measured_at"]), name
        assert host["rustc"].startswith("rustc 1.98.1 "), name


def test_the_literal_millisecond_deadlines_left_are_the_known_ones() -> None:
    """No new literal millisecond limit enters the engine; a converted module leaves this list."""
    found: dict[str, int] = {}
    for source in sorted(ENGINE_SOURCES.rglob("*.rs")):
        count = len(_LITERAL_DEADLINE.findall(source.read_text(encoding="utf-8")))
        if count:
            found[str(source.relative_to(REPO_ROOT))] = count
    assert found == _LITERAL_DEADLINES_LEFT


def _write_engine(root: Path, files: dict[str, str]) -> Path:
    """Write a small engine source tree under ``root`` and return its ``engine/src`` folder.

    Parameters
    ----------
    root : Path
        The folder that stands for the repository root.
    files : dict[str, str]
        Source text by path relative to ``engine/src``.
    """
    sources = root / "engine" / "src"
    for name, text in files.items():
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return sources


_BUDGET = 'host_budget::assert_within(module_path!(), "{name}", || {{}});'


def test_a_budget_in_a_path_included_tests_file_is_keyed_by_its_parent_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``#[path = "<name>_tests.rs"] mod tests;`` file holds its parent's tests; its budgets are keyed as such.

    ``module_path!()`` in such a file is ``<parent>::tests``, so the profile key is the parent module's path plus
    ``tests``, not a path built from the included file's name.
    """
    sources = _write_engine(
        tmp_path,
        {
            "neurons/population/brunel.rs": (
                'pub struct B;\n\n#[cfg(test)]\n#[path = "brunel_tests.rs"]\nmod tests;\n'
            ),
            "neurons/population/brunel_tests.rs": (
                "use super::*;\n\n#[test]\nfn brunel_performance() {\n    "
                + _BUDGET.format(name="brunel_performance")
                + "\n}\n"
            ),
            "neurons/motor/alpha.rs": (
                "pub struct A;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn alpha_performance() {\n        "
                + _BUDGET.format(name="alpha_performance")
                + "\n    }\n}\n"
            ),
        },
    )
    monkeypatch.setitem(globals(), "ENGINE_SOURCES", sources)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    found = {
        key: (source, function, "#[test]" in attributes)
        for source, key, function, attributes in _budgeted_tests()
    }
    assert found == {
        "neurons::motor::alpha::tests::alpha_performance": (
            "engine/src/neurons/motor/alpha.rs",
            "alpha_performance",
            True,
        ),
        "neurons::population::brunel::tests::brunel_performance": (
            "engine/src/neurons/population/brunel_tests.rs",
            "brunel_performance",
            True,
        ),
    }


@pytest.mark.parametrize(
    "files",
    [
        {"neurons/x.rs": "pub fn f() {\n    " + _BUDGET.format(name="f") + "\n}\n"},
        {
            "neurons/x.rs": '#[path = "x_helpers.rs"]\nmod helpers;\n',
            "neurons/x_helpers.rs": "pub fn f() {\n    " + _BUDGET.format(name="f") + "\n}\n",
        },
    ],
    ids=["inline, outside mod tests", "path-included, a module other than tests"],
)
def test_a_budget_outside_a_tests_module_is_refused(
    files: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget belongs to a test, so one outside ``mod tests``, inline or included by ``#[path]``, is refused.

    Parameters
    ----------
    files : dict[str, str]
        Source text by path relative to ``engine/src``.
    """
    monkeypatch.setitem(globals(), "ENGINE_SOURCES", _write_engine(tmp_path, files))
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="a budget outside `mod tests`"):
        _budgeted_tests()


def _budget_keys(
    files: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Return each budget's key with its source file, for a small engine tree.

    Parameters
    ----------
    files : dict[str, str]
        Source text by path relative to ``engine/src``.
    tmp_path : Path
        The folder that stands for the repository root.
    monkeypatch : pytest.MonkeyPatch
        Points the guard at that tree for the calling test.
    """
    monkeypatch.setitem(globals(), "ENGINE_SOURCES", _write_engine(tmp_path, files))
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    return {key: source for source, key, _function, _attributes in _budgeted_tests()}


def _inline_tests(name: str) -> str:
    """Return an inline ``mod tests`` block holding one budgeted test called ``name``.

    Parameters
    ----------
    name : str
        The test function's name, which its budget also carries.
    """
    return (
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn "
        + name
        + "() {\n        "
        + _BUDGET.format(name=name)
        + "\n    }\n}\n"
    )


@pytest.mark.parametrize("visibility", ["", "pub ", "pub(crate) ", "pub(in crate::neurons) "])
def test_a_path_included_module_keys_its_own_inline_tests_below_it(
    visibility: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``#[path]`` file included as a module other than ``tests`` keys its inline tests below that module.

    ``module_path!()`` in the inline ``mod tests`` of a file included by ``#[path = "x_helpers.rs"] mod helpers;`` is
    ``neurons::x::helpers::tests``. A visibility before ``mod`` leaves the path as it is.

    Parameters
    ----------
    visibility : str
        What stands before ``mod`` in the declaration.
    """
    keys = _budget_keys(
        {
            "neurons/x.rs": f'#[path = "x_helpers.rs"]\n{visibility}mod helpers;\n',
            "neurons/x_helpers.rs": "pub fn g() {}\n\n" + _inline_tests("f"),
        },
        tmp_path,
        monkeypatch,
    )
    assert keys == {"neurons::x::helpers::tests::f": "engine/src/neurons/x_helpers.rs"}


def test_a_module_path_follows_path_declarations_back_to_the_crate_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file included by ``#[path]`` from a file that is itself included so takes its declarer's module path.

    The engine has such chains: ``pyo3_neurons.rs`` includes ``bindings/biophysical/mod.rs``, which includes
    ``bindings/glif.rs``. The crate root ``lib.rs`` has the empty path, as ``module_path!()`` without the crate name.
    """
    keys = _budget_keys(
        {
            "lib.rs": '#[path = "bindings/outer.rs"]\nmod outer_binding;\n\n'
            + _inline_tests("root"),
            "bindings/outer.rs": '#[path = "inner_impl.rs"]\nmod inner;\n',
            "bindings/inner_impl.rs": _inline_tests("inner"),
        },
        tmp_path,
        monkeypatch,
    )
    assert keys == {
        "tests::root": "engine/src/lib.rs",
        "outer_binding::inner::tests::inner": "engine/src/bindings/inner_impl.rs",
    }


def test_a_tests_file_declared_without_path_is_keyed_by_its_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mod tests;`` in ``neuron/adex.rs`` loads ``neuron/adex/tests.rs``, whose module path ends in ``tests``.

    The AdEx tests keep a literal deadline in such a file, so converting them needs the guard to read it.
    """
    keys = _budget_keys(
        {
            "neuron/adex.rs": "pub struct A;\n\n#[cfg(test)]\nmod tests;\n",
            "neuron/adex/tests.rs": (
                "use super::*;\n\n#[test]\nfn adex_performance() {\n    "
                + _BUDGET.format(name="adex_performance")
                + "\n}\n"
            ),
        },
        tmp_path,
        monkeypatch,
    )
    assert keys == {"neuron::adex::tests::adex_performance": "engine/src/neuron/adex/tests.rs"}

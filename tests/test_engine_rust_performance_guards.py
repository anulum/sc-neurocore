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

# Literal millisecond limits in modules whose documentation units were still open when the first modules were
# converted. Each later unit converts a module's tests and removes its line here.
_LITERAL_DEADLINES_LEFT = {
    "engine/src/neuron/adex/tests.rs": 1,
    "engine/src/neuron/exp_if_tests.rs": 1,
    "engine/src/neurons/ermentrout_kopell_map.rs": 1,
    "engine/src/neurons/motor/alpha_motor_neuron.rs": 1,
    "engine/src/neurons/motor/gamma_motor_neuron.rs": 1,
    "engine/src/neurons/motor/motor_unit.rs": 1,
    "engine/src/neurons/motor/renshaw_cell.rs": 1,
    "engine/src/neurons/motor/upper_motor_neuron.rs": 1,
    "engine/src/neurons/population/brunel_network_tests.rs": 1,
    "engine/src/neurons/population/montbrio_mean_field_tests.rs": 1,
    "engine/src/neurons/population/tum_network_tests.rs": 1,
}


def _profiles() -> dict[str, object]:
    """Read ``engine/test_host_profiles.toml``."""
    return tomllib.loads(PROFILES.read_text(encoding="utf-8"))


def _module_path(source: Path) -> str:
    """Return the Rust module path of an engine source file, without the crate name."""
    parts = list(source.relative_to(ENGINE_SOURCES).with_suffix("").parts)
    if parts[-1] == "mod":
        parts.pop()
    return "::".join(parts)


def _budgeted_tests() -> list[tuple[str, str, str, str]]:
    """Find every budgeted test as (source, key, enclosing function, attributes above that function)."""
    found = []
    for source in sorted(ENGINE_SOURCES.rglob("*.rs")):
        text = source.read_text(encoding="utf-8")
        tests_module = text.find("mod tests {")
        for call in _CALL.finditer(text):
            assert tests_module != -1 and tests_module < call.start(), (
                f"{source}: a budget outside `mod tests`"
            )
            functions = list(_FUNCTION.finditer(text, 0, call.start()))
            enclosing = functions[-1]
            line_start = text.rfind("\n", 0, enclosing.start())
            previous_end = text.rfind("}", 0, line_start)
            attributes = text[previous_end + 1 : line_start]
            key = f"{_module_path(source)}::tests::{call['name']}"
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
    """The first modules converted hold 29 budgeted tests; a test is not silently dropped."""
    assert len(_budgeted_tests()) == 29


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

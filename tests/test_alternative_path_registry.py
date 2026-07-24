# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (registry) from former test_alternative_path.py

from __future__ import annotations

from tests.alternative_path_support import *  # noqa: F403


def test_registry_register_get_and_run():
    registry = AlternativePathRegistry()
    route = AlternativePathRoute(
        name="safe.registry",
        baseline=lambda x: x * 2,
        candidate=lambda x: x * 3,
        summary="Registry route",
        expected_behavior="Candidate can be enabled by name",
    )
    registry.register(route)

    assert registry.names() == ("safe.registry",)
    assert registry.get("safe.registry") is route
    result = registry.run("safe.registry", AlternativePathConfig(), 5)
    assert result.value == 10


def test_registry_rejects_duplicate_names():
    registry = AlternativePathRegistry()
    route = AlternativePathRoute(
        name="safe.dup",
        baseline=lambda: 1,
        candidate=lambda: 2,
        summary="Duplicate route",
        expected_behavior="Registry must reject duplicate names",
    )
    registry.register(route)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(route)


def test_registry_describe_returns_metadata():
    registry = build_demo_registry()

    descriptions = registry.describe()

    assert len(descriptions) == 1
    assert descriptions[0]["name"] == "demo.affine-sigmoid"
    assert "vectorised NumPy sigmoid candidate" in descriptions[0]["summary"]


def test_demo_registry_evaluate_matches_on_demo_route():
    registry = build_demo_registry()

    summary = registry.evaluate(
        "demo.affine-sigmoid",
        [
            AlternativePathCase("v1", args=([0.0, 1.0, -1.0],)),
            AlternativePathCase("v2", args=([2.0, -2.0],), kwargs={"bias": 0.25}),
        ],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )

    assert summary.total_cases == 2
    assert summary.matched_cases == 2
    assert summary.candidate_failures == 0
    assert summary.median_baseline_runtime_ns is not None
    assert summary.median_candidate_runtime_ns is not None


def test_builtin_registry_exposes_real_physics_route():
    registry = build_builtin_registry()

    descriptions = registry.describe()
    names = {item["name"] for item in descriptions}

    assert "memory.delayed-recall.shared-state" in names
    assert "physics.heat.cosine-mode" in names
    assert "physics.oscillator.harmonic-symplectic" in names
    assert "physics.kuramoto.noiseless-symplectic-lift" in names
    assert "solver.lif.subthreshold-exact" in names


def test_builtin_cases_cover_shared_state_route_and_reject_unknown() -> None:
    cases = builtin_cases_for_route("memory.delayed-recall.shared-state")
    assert len(cases) >= 1

    with pytest.raises(KeyError, match="No built-in cases for route"):
        builtin_cases_for_route("route.that.does.not.exist")


def test_registry_get_unknown_route_raises_keyerror():
    registry = AlternativePathRegistry()
    with pytest.raises(KeyError, match="Unknown alternative path"):
        registry.get("missing")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A catalogue model compiles with the parameters the Studio shows

"""The compile translates a catalogue model's parameters into its schema's, and the RTL is the same neuron.

The Studio shows and sends a model's catalogue parameters. Nine models name
some of them differently from their canonical schema, so 'Generate RTL'
refused them as unknown overrides. Each translation is proven here the only
way that counts: the catalogue class and the schema neuron the compiler builds
from the translated request are run side by side at values other than the
defaults, and they follow the same trajectory.
"""

from collections.abc import Callable, Mapping
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import (
    ResolvedModelCompileConfiguration,
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_compile_parameters import (
    DERIVED,
    RENAMED,
    SEEDED,
    CatalogueParametersRefused,
    SchemaParameters,
    to_schema_parameters,
)
from sc_neurocore.studio.platform import StudioRuntimeSettings

#: The nine models whose catalogue parameters the compile refused as unknown before this change.
TRANSLATED = (
    "AdExNeuron",
    "AmariNeuralField",
    "BrunelWangNeuron",
    "EscapeRateNeuron",
    "HodgkinHuxleyNeuron",
    "JansenRitUnit",
    "PoissonNeuron",
    "ResonateAndFireNeuron",
    "WangBuzsakiNeuron",
)


def studio_request(model_name: str, **changed: float) -> dict[str, object]:
    """Build the compile request the Studio sends, with some catalogue values changed.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    **changed : float
        Catalogue parameters set to other values than their defaults.

    Returns
    -------
    dict[str, object]
        The catalogue parameters, the model's own step, and its default integrator and Q-format.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    configuration = detail["compile_configuration"]
    params = {parameter["name"]: parameter["default"] for parameter in detail["params"]}
    unknown = set(changed) - set(params)
    assert not unknown, unknown
    params.update(changed)
    return {
        "model_name": model_name,
        "dt": detail["dt"],
        "integrator": configuration["default_integrator"],
        "q_format": configuration["default_q_format"],
        "params": params,
    }


def resolve(model_name: str, **changed: float) -> ResolvedModelCompileConfiguration:
    """Resolve the Studio's request as the compiler does.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    **changed : float
        Catalogue parameters set to other values than their defaults.

    Returns
    -------
    ResolvedModelCompileConfiguration
        What the compiler would build.
    """
    return resolve_model_compile_configuration(
        studio_request(model_name, **changed), detail_getter=get_model_detail
    )


def test_the_declared_names_are_the_catalogue_s_and_the_schema_s() -> None:
    """Every name the table translates exists on both sides, so the table cannot drift unseen."""
    assert set(RENAMED) | set(DERIVED) | set(SEEDED) == set(TRANSLATED)
    for model_name, renames in RENAMED.items():
        detail = get_model_detail(model_name)
        assert detail is not None
        schema = load_schema(detail["compile_configuration"]["schema_name"])
        catalogue = {parameter["name"] for parameter in detail["params"]}
        assert set(renames) <= catalogue - set(schema["parameters"]), model_name
        assert set(renames.values()) <= set(schema["parameters"]), model_name
    for model_name in SEEDED:
        detail = get_model_detail(model_name)
        assert detail is not None
        assert "seed" in {parameter["name"] for parameter in detail["params"]}
        assert detail["profile_contract"]["numerical"]["randomness"]["kind"] == "lfsr16-threshold"


@pytest.mark.parametrize("model_name", [name for name in TRANSLATED if name != "AmariNeuralField"])
def test_each_translated_model_compiles_at_the_studio_s_defaults(model_name: str) -> None:
    """At the catalogue defaults the translation lands on the schema's own defaults.

    Parameters
    ----------
    model_name : str
        One of the eight models whose default request now compiles.
    """
    resolved = resolve(model_name)
    detail = get_model_detail(model_name)
    assert detail is not None
    schema = load_schema(detail["compile_configuration"]["schema_name"])
    for name, value in resolved.params.items():
        assert name in schema["parameters"], name
        assert value == pytest.approx(schema["parameters"][name], rel=1e-15, abs=1e-15), name
    assert "module sc_" in resolved.to_verilog()


def test_the_amari_field_compiles_only_as_the_schema_s_four_sites() -> None:
    """The RTL is the schema's fixed four-site field: n = 64 is refused in words, n = 4 compiles."""
    with pytest.raises(CatalogueParametersRefused) as refused:
        resolve("AmariNeuralField")
    assert isinstance(refused.value, AuthoredRefusal)
    assert str(refused.value) == (
        "AmariNeuralField compiles to the schema's fixed field of 4 sites; n = 64 cannot be "
        "compiled. Set n to 4."
    )
    assert "module sc_" in resolve("AmariNeuralField", n=4).to_verilog()


def _pairs(
    hand: Any,
    neuron: UniversalNeuron,
    steps: int,
    drive: Callable[[int], float],
    hand_state: Callable[[Any], tuple[float, ...]],
    keys: tuple[str, ...],
) -> float:
    """Run both side by side, check that their events agree, and return the largest state difference.

    Parameters
    ----------
    hand : Any
        The catalogue class instance.
    neuron : UniversalNeuron
        The neuron the compiler built from the translated request.
    steps : int
        How many steps to run.
    drive : Callable[[int], float]
        The input at each step.
    hand_state : Callable[[Any], tuple[float, ...]]
        Reads the class's state in the order of ``keys``.
    keys : tuple of str
        The schema's state names.

    Returns
    -------
    float
        The largest absolute state difference over the run.
    """
    worst = 0.0
    for index in range(steps):
        assert int(neuron.step(I=drive(index))) == int(hand.step(drive(index))), index
        expected = hand_state(hand)
        actual = tuple(neuron.state[key] for key in keys)
        worst = max(worst, *(abs(a - b) for a, b in zip(actual, expected, strict=True)))
    return worst


def _sine(base: float, amplitude: float) -> Callable[[int], float]:
    """Return a slowly varying drive.

    Parameters
    ----------
    base : float
        Its mean.
    amplitude : float
        Its swing.

    Returns
    -------
    Callable[[int], float]
        The input at each step.
    """
    return lambda index: base + amplitude * math.sin(index * 0.037)


HH_CHANGED = {
    "c_m": 1.3,
    "g_na": 100.0,
    "g_k": 30.0,
    "g_l": 0.4,
    "e_na": 55.0,
    "e_k": -80.0,
    "e_l": -60.0,
}
WB_CHANGED = {
    "c_m": 1.2,
    "g_na": 30.0,
    "g_k": 10.0,
    "g_l": 0.15,
    "e_na": 50.0,
    "e_k": -85.0,
    "e_l": -63.0,
}


@pytest.mark.parametrize(
    ("model_name", "changed", "hand", "steps", "drive", "keys", "tolerance"),
    [
        (
            "AdExNeuron",
            {"b": 30.0, "c_m": 150.0, "delta_t": 3.0},
            lambda changed: AdExNeuron(**changed),
            1_000,
            _sine(500.0, 200.0),
            ("v", "w"),
            1e-12,
        ),
        (
            "ResonateAndFireNeuron",
            {"threshold": 0.6},
            lambda changed: ResonateAndFireNeuron(**changed),
            400,
            _sine(4.0, 1.2),
            ("x", "y"),
            1e-13,
        ),
        (
            "HodgkinHuxleyNeuron",
            HH_CHANGED,
            lambda changed: HodgkinHuxleyNeuron(**changed, integrator="rk4"),
            200,
            _sine(10.0, 4.0),
            ("v", "m", "h", "n"),
            1e-9,
        ),
        (
            "WangBuzsakiNeuron",
            WB_CHANGED,
            lambda changed: WangBuzsakiNeuron(**changed),
            200,
            _sine(10.0, 4.0),
            ("v", "h", "n"),
            1e-9,
        ),
    ],
)
def test_a_renamed_parameter_is_the_same_parameter(
    model_name: str,
    changed: Mapping[str, float],
    hand: Callable[[Mapping[str, float]], Any],
    steps: int,
    drive: Callable[[int], float],
    keys: tuple[str, ...],
    tolerance: float,
) -> None:
    """Off the defaults, the class and the compiled schema neuron follow one trajectory and one spike train.

    Parameters
    ----------
    model_name : str
        A model whose catalogue names are renamed for the schema.
    changed : Mapping[str, float]
        Catalogue values other than the defaults, every renamed one among them.
    hand : Callable
        Builds the catalogue class with those values.
    steps : int
        How many steps to compare.
    drive : Callable[[int], float]
        The input at each step.
    keys : tuple of str
        The state compared, by its schema names.
    tolerance : float
        The largest state difference admitted.
    """
    assert set(RENAMED[model_name]) <= set(changed)
    neuron = resolve(model_name, **changed).neuron
    instance = hand(changed)
    worst = _pairs(
        instance, neuron, steps, drive, lambda h: tuple(getattr(h, key) for key in keys), keys
    )
    assert worst <= tolerance


def test_the_jansen_rit_connectivity_derives_its_four_couplings() -> None:
    """``c`` becomes C1 = c, C2 = 0.8 c and C3 = C4 = 0.25 c, as the class and the schema both state."""
    resolved = resolve("JansenRitUnit", c=120.0)
    assert {key: resolved.params[key] for key in ("c1", "c2", "c3", "c4")} == pytest.approx(
        {"c1": 120.0, "c2": 96.0, "c3": 30.0, "c4": 30.0}, rel=1e-15
    )
    neuron = resolved.neuron
    hand = JansenRitUnit(c=120.0)
    keys = ("y0", "y3", "y1", "y4", "y2", "y5")
    worst = 0.0
    for index in range(400):
        drive = 220.0 + 100.0 * math.sin(index * 0.037)
        hand.step(drive)
        neuron.step(I=drive)
        worst = max(worst, *(abs(neuron.state[key] - getattr(hand, key)) for key in keys))
    assert worst <= 1e-12


def test_the_amari_kernel_is_sampled_from_its_catalogue_parameters() -> None:
    """The kernel's four parameters and ``dx`` become the schema's sampled weights w0, w1 and w2."""
    changed = {"n": 4, "a_exc": 1.8, "a_width": 1.5, "b_inh": 0.9, "b_width": 0.8}
    resolved = resolve("AmariNeuralField", **changed)
    for k in range(3):
        distance = k * 0.5
        expected = 1.8 * math.exp(-1.5 * distance) - 0.9 * math.exp(-0.8 * distance)
        assert resolved.params[f"w{k}"] == pytest.approx(expected, rel=1e-15)
    neuron = resolved.neuron
    hand = AmariNeuralField(n=4, a_exc=1.8, a_width=1.5, b_inh=0.9, b_width=0.8)
    for current in (0.1, -0.2, 0.3, 0.05, -0.1) * 20:
        hand.step(current)
        neuron.step(I=current)
        assert hand.u is not None
        np.testing.assert_allclose(
            np.asarray(list(neuron.state.values()), dtype=np.float64), hand.u, rtol=0.0, atol=1e-15
        )


def test_the_brunel_wang_capacitance_is_the_same_capacitance() -> None:
    """``C_m`` is the schema's ``c_m``: off its default, voltage and events stay with the class."""
    neuron = resolve("BrunelWangNeuron", C_m=0.4).neuron
    hand = BrunelWangNeuron(C_m=0.4)
    for index in range(256):
        gates = (
            0.035 + 0.018 * (1.0 + math.sin(index * 0.071)),
            0.12 + 0.05 * (1.0 + math.cos(index * 0.053)),
            0.08 + 0.04 * (1.0 + math.sin(index * 0.037 + 0.2)),
            0.03 + 0.02 * (1.0 + math.cos(index * 0.089)),
        )
        event = hand.step(*gates)
        # The schema's lowering takes the four gates and three empty edges per physical step.
        for value in (*gates, 0.0, 0.0, 0.0):
            neuron.step(I=value)
        assert int(neuron.state["spike_flag"]) == event
        assert neuron.state["v"] == pytest.approx(hand.v, abs=2e-12)


@pytest.mark.parametrize("seed", [44257, 1234])
def test_the_seed_is_the_schema_s_lfsr_seed(seed: int) -> None:
    """The catalogue ``seed`` seeds the schema's LFSR16: the event trains are identical.

    Parameters
    ----------
    seed : int
        The default seed and another.
    """
    escape = resolve("EscapeRateNeuron", seed=float(seed))
    assert escape.rng_seed == seed
    hand_escape = EscapeRateNeuron(seed=seed)
    assert [int(escape.neuron.step(I=18.0)) for _ in range(2_000)] == [
        int(hand_escape.step(18.0)) for _ in range(2_000)
    ]
    poisson = resolve("PoissonNeuron", seed=float(seed), rate_hz=150.0)
    hand_poisson = PoissonNeuron(seed=seed, rate_hz=150.0)
    # The schema reads its input as a rate override; a negative one keeps rate_hz, as the class's default does.
    assert [int(poisson.neuron.step(I=-1.0)) for _ in range(2_000)] == [
        int(hand_poisson.step()) for _ in range(2_000)
    ]
    assert f"16'h{seed:04x}" in escape.to_verilog()


@pytest.mark.parametrize(
    ("seed", "reason"),
    [
        (1.5, "EscapeRateNeuron's seed must be a whole number; received 1.5."),
        (70_000.0, "EscapeRateNeuron accepts a seed in [0, 65535]; received 70000"),
    ],
)
def test_a_seed_the_lfsr_cannot_take_is_refused_in_words(seed: float, reason: str) -> None:
    """A seed outside the LFSR16 state space is the caller's input, refused before any RTL.

    Parameters
    ----------
    seed : float
        A fractional seed, and one past the sixteen-bit range.
    reason : str
        The whole refusal.
    """
    with pytest.raises(CatalogueParametersRefused) as refused:
        resolve("EscapeRateNeuron", seed=seed)
    assert str(refused.value) == reason


def test_a_catalogue_name_and_its_schema_name_together_are_refused() -> None:
    """Two names for one parameter in one request cannot both be honoured, so neither is guessed."""
    request = studio_request("AdExNeuron")
    params = request["params"]
    assert isinstance(params, dict)
    params["b_adapt"] = 9.0
    with pytest.raises(CatalogueParametersRefused) as refused:
        resolve_model_compile_configuration(request, detail_getter=get_model_detail)
    assert str(refused.value) == (
        "AdExNeuron received both 'b' and its schema name 'b_adapt'; send one of them."
    )


def test_a_derived_value_sent_beside_its_source_is_refused() -> None:
    """A coupling sent beside the ``c`` it is derived from would be overwritten, so it is refused."""
    request = studio_request("JansenRitUnit")
    params = request["params"]
    assert isinstance(params, dict)
    params["c2"] = 100.0
    with pytest.raises(CatalogueParametersRefused) as refused:
        resolve_model_compile_configuration(request, detail_getter=get_model_detail)
    assert str(refused.value) == (
        "JansenRitUnit received 'c2', which it derives from its catalogue parameters; "
        "send those instead."
    )


def test_a_name_neither_side_knows_is_still_refused() -> None:
    """The translation names what it knows; anything else stays an unknown override."""
    request = studio_request("AdExNeuron")
    params = request["params"]
    assert isinstance(params, dict)
    params["bogus"] = 1.0
    with pytest.raises(AuthoredRefusal, match=r"^Unknown schema parameter override\(s\): bogus$"):
        resolve_model_compile_configuration(request, detail_getter=get_model_detail)


def test_the_studio_s_adex_request_compiles_over_http(tmp_path: Path) -> None:
    """The request the Studio sends for AdEx is answered with RTL, and the record names the schema's values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/models/compile",
            json=studio_request("AdExNeuron", b=30.0),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "module sc_" in body["verilog"]
        assert body["compile_configuration"]["rng_seed"] is None
        params = body["compile_traceability"]["source_payload"]["params"]
        assert params["b_adapt"] == 30.0
        assert "b" not in params


def test_a_request_without_the_derived_sources_keeps_the_schema_s_values() -> None:
    """A caller that sends neither ``c`` nor a kernel parameter gets the schema's own couplings and weights."""
    jansen = to_schema_parameters(
        "JansenRitUnit", {"e0": 2.5}, catalogue_defaults={"c": 135.0}, schema={}
    )
    assert jansen.parameters == {"e0": 2.5}
    detail = get_model_detail("AmariNeuralField")
    assert detail is not None
    schema = load_schema(detail["compile_configuration"]["schema_name"])
    amari = to_schema_parameters(
        "AmariNeuralField", {"tau": 12.0}, catalogue_defaults={}, schema=schema
    )
    assert amari == SchemaParameters(parameters={"tau": 12.0}, rng_seed=None)


def test_a_schema_that_states_no_number_of_sites_is_a_server_fault() -> None:
    """Without its ``site_count`` the field cannot be checked; that is the server's schema, not the caller."""
    with pytest.raises(ValueError, match="states no integer site_count") as fault:
        to_schema_parameters("AmariNeuralField", {"n": 4.0}, catalogue_defaults={}, schema={})
    assert not isinstance(fault.value, AuthoredRefusal)


@pytest.mark.parametrize(
    ("params", "fault"),
    [
        ("not a list", "Studio model detail 'params' must be a list."),
        ([{"default": 1.0}], "Studio model detail parameters must be objects with a name."),
    ],
)
def test_a_catalogue_detail_in_another_shape_is_a_server_fault(params: object, fault: str) -> None:
    """The catalogue's own record, not the request, is wrong: a plain error the job will record.

    Parameters
    ----------
    params : object
        The detail's parameter list, malformed.
    fault : str
        The whole message.
    """
    detail = get_model_detail("AdExNeuron")
    assert detail is not None
    broken = {**detail, "params": params}
    with pytest.raises(ValueError) as raised:
        resolve_model_compile_configuration(
            studio_request("AdExNeuron"), detail_getter=lambda _name: broken
        )
    assert str(raised.value) == fault
    assert not isinstance(raised.value, AuthoredRefusal)


def test_a_kernel_parameter_without_a_catalogue_default_is_a_server_fault() -> None:
    """A listed parameter with no numeric default gives none; a derivation that needs it names the record."""
    detail = get_model_detail("AmariNeuralField")
    assert detail is not None
    listed = [
        {**parameter, "default": None} if parameter["name"] == "b_width" else parameter
        for parameter in detail["params"]
    ]
    request = studio_request("AmariNeuralField", n=4, a_exc=1.8)
    params = request["params"]
    assert isinstance(params, dict)
    del params["b_width"]
    with pytest.raises(ValueError) as raised:
        resolve_model_compile_configuration(
            request, detail_getter=lambda _name: {**detail, "params": listed}
        )
    assert str(raised.value) == (
        "Studio model 'AmariNeuralField' has no catalogue default for 'b_width'."
    )
    assert not isinstance(raised.value, AuthoredRefusal)


def test_a_request_in_the_schema_s_own_names_is_honoured_as_before() -> None:
    """An API caller that already used the schema's names keeps getting exactly what it asked for."""
    resolved = resolve_model_compile_configuration(
        {
            "model_name": "AdExNeuron",
            "dt": 0.1,
            "q_format": "Q16.16",
            "params": {"b_adapt": 30.0, "C": 150.0},
        },
        detail_getter=get_model_detail,
    )
    assert resolved.params == {"b_adapt": 30.0, "C": 150.0}
    assert resolved.rng_seed is None


@pytest.mark.parametrize(
    "changed",
    [
        {"n": 4, "dx": 1.0},
        {"n": 4, "dx": 0.8, "a_exc": 1.8, "b_width": 0.7},
    ],
)
def test_the_amari_kernel_is_sampled_at_the_request_s_dx(changed: dict[str, float]) -> None:
    """A request in catalogue names samples the kernel at its own ``dx``, as the class does.

    ``n`` and the kernel's four are catalogue names the schema does not have,
    so a request that carries one of them is read in catalogue names: the
    weights are derived at the request's ``dx`` even when it carries no
    kernel parameter, rather than kept at the schema's 0.5.

    Parameters
    ----------
    changed : dict of str to float
        The request's parameters, at a ``dx`` other than 0.5.
    """
    detail = get_model_detail("AmariNeuralField")
    assert detail is not None
    resolved = resolve_model_compile_configuration(
        {
            "model_name": "AmariNeuralField",
            "dt": detail["dt"],
            "q_format": detail["compile_configuration"]["default_q_format"],
            "params": changed,
        },
        detail_getter=get_model_detail,
    )
    kernel: dict[str, Any] = {"a_exc": 1.5, "a_width": 2.0, "b_inh": 0.75, "b_width": 1.0}
    kernel.update({key: value for key, value in changed.items() if key != "n"})
    for k in range(3):
        distance = k * kernel["dx"]
        expected = kernel["a_exc"] * math.exp(-kernel["a_width"] * distance) - kernel[
            "b_inh"
        ] * math.exp(-kernel["b_width"] * distance)
        assert resolved.params[f"w{k}"] == pytest.approx(expected, rel=1e-15)
    hand = AmariNeuralField(n=4, **kernel)
    neuron = resolved.neuron
    for current in (0.1, -0.2, 0.3, 0.05, -0.1) * 20:
        hand.step(current)
        neuron.step(I=current)
        assert hand.u is not None
        np.testing.assert_allclose(
            np.asarray(list(neuron.state.values()), dtype=np.float64), hand.u, rtol=0.0, atol=1e-15
        )


@pytest.mark.parametrize("seed", [44257, 1234])
def test_the_seed_is_part_of_the_traceability_input(tmp_path: Path, seed: int) -> None:
    """The traceability's declared input names the seed, so two seeds never share an input digest.

    The seed changes the RTL (its LFSR starts there), so an input that left it
    out would no longer determine its output. A deterministic model's input,
    and so its digest, is unchanged: it carries no seed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned job directory.
    seed : int
        The default seed and another.
    """
    import threading

    from sc_neurocore.studio.platform.jobs import StudioJobContext
    from sc_neurocore.studio.platform.model_compile_process import run_model_compile_process_task

    def traceability(job: str, request: dict[str, object]) -> dict[str, Any]:
        """Compile one request in its own job directory and return its traceability.

        Parameters
        ----------
        job : str
            The job's directory name.
        request : dict of str to object
            The compile request.

        Returns
        -------
        dict[str, Any]
            The compile traceability.
        """
        work_dir = tmp_path / job
        work_dir.mkdir()
        context = StudioJobContext(
            job_id=f"sj_{job}",
            work_dir=work_dir,
            cancel_event=threading.Event(),
            max_artifact_bytes=2 * 1024 * 1024,
        )
        result = run_model_compile_process_task(context, request)
        traced = result["compile_traceability"]
        assert isinstance(traced, dict)
        return traced

    seeded = traceability("seeded", studio_request("EscapeRateNeuron", seed=float(seed)))
    other = traceability("other", studio_request("EscapeRateNeuron", seed=float(seed + 1)))
    assert seeded["source_payload"]["rng_seed"] == seed
    assert seeded["input_sha256"] != other["input_sha256"]
    plain = traceability("plain", {"model_name": "SCLapicqueLIFNeuron", "q_format": "Q8.8"})
    assert "rng_seed" not in plain["source_payload"]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed default-neuron PyO3 contracts

"""Exercise the shared default-neuron binding through every actual registered class."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import pickle
import re
import sys
from typing import Protocol, cast
import unittest

import numpy as np

_NATIVE = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
_NAMESPACE = "sc_neurocore_engine.sc_neurocore_engine"
_DEFAULT_NEURONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ATypeKNeuron", ("v", "h", "n", "a", "b")),
    ("AdaptiveThresholdIFNeuron", ("v", "theta")),
    ("AlphaMotorNeuron", ("v", "h", "n", "m_pic", "ca")),
    ("ArcaneNeuron", ("v_fast", "v_work", "v_deep")),
    ("AttentionGatedNeuron", ("v",)),
    ("AvRonCardiacNeuron", ("v", "h", "n", "s")),
    ("BKNeuron", ("v", "h", "n", "ca")),
    ("BalancedResonateAndFireNeuron", ("x", "y", "q")),
    ("BertramPhantomBurster", ("v", "n", "s1", "s2")),
    ("BoothRinzelNeuron", ("vs", "vd", "ca")),
    ("BrainScaleSAdExNeuron", ("v", "w")),
    ("BrunelNetwork", ("r_e", "r_i")),
    ("ButeraRespiratoryNeuron", ("v", "n", "h_nap")),
    ("CardiacPurkinjeFibre", ("v", "d", "f", "y")),
    ("CazellesMapNeuron", ("x",)),
    ("CerebellarBasketNeuron", ("v", "h", "n", "a", "b", "ca")),
    ("ChandelierNeuron", ("v", "h", "n", "d", "p")),
    ("ChayKeizerNeuron", ("v", "m", "h", "n", "ca")),
    ("ChayNeuron", ("v", "n", "ca")),
    ("ChialvoMapNeuron", ("x", "y")),
    ("ClosedFormContinuousNeuron", ("x",)),
    ("ComplementaryLIFNeuron", ("v_pos", "v_neg")),
    ("CompositionalBindingNeuron", ("phi", "amplitude")),
    ("ConnorStevensNeuron", ("v", "m", "h", "n", "a", "b")),
    ("CourageNekorkinMapNeuron", ("x", "y")),
    ("DCNNeuron", ("v", "h", "n", "p", "s", "r", "ca")),
    ("DPINeuron", ("i_mem", "i_ahp", "refractory_time")),
    ("DeSchutterPurkinjeNeuron", ("v", "h_na", "n_k", "m_cap", "h_cap", "q_kca", "ca")),
    ("DendrifyNeuron", ("v_s", "v_d")),
    ("DestexheThalamicNeuron", ("v", "h_na", "n_k", "m_t", "h_t")),
    ("DifferentiableSurrogateNeuron", ("v",)),
    ("DurstewitzDopamineNeuron", ("v", "h_na", "n_k")),
    ("ElBoustaniNetwork", ("r_e", "r_i", "s")),
    ("EndocrineBetaCell", ("v", "n", "ca")),
    ("ErmentroutKopellMapNeuron", ("theta",)),
    ("FitzHughNagumoNeuron", ("v", "w")),
    ("FitzHughRinzelNeuron", ("v", "w", "y")),
    ("FrankenhaeUserHuxleyAxon", ("v", "m", "h", "n", "p")),
    (
        "GLIFNeuron",
        ("v", "theta_spike", "i_asc1", "i_asc2", "theta_voltage", "refractory_remaining"),
    ),
    ("GammaMotorNeuron", ("v", "adapt")),
    ("GapJunctionNeuron", ("v",)),
    ("GatedLIFNeuron", ("v",)),
    ("GolgiCell", ("v", "m", "h", "p_na", "n", "a", "b", "w", "m_t", "s", "c_n", "r", "ca")),
    ("GolombFSNeuron", ("v", "h", "n", "p")),
    ("GradedSynapseNeuron", ("v",)),
    ("GranuleCell", ("v", "m", "h", "n", "ca")),
    ("GutkinErmentroutNeuron", ("v", "n")),
    ("HillTononiNeuron", ("v", "theta", "d_k", "m_h", "m_t", "h_t", "spike_timer")),
    ("HindmarshRoseNeuron", ("x", "y", "z")),
    ("HodgkinHuxleyNeuron", ("v", "m", "h", "n")),
    ("HuberBraunNeuron", ("v", "a_sd", "a_sr")),
    ("IbarzTanakaMapNeuron", ("v", "u")),
    ("InhibitoryLIFNeuron", ("v", "inh_trace")),
    ("KLIFNeuron", ("v",)),
    ("LearnableNeuronModel", ("v",)),
    ("LiquidTimeConstantNeuron", ("x",)),
    ("LugaroCell", ("v", "adapt")),
    ("MarderSTGNeuron", ("v", "ca")),
    ("MartinottiNeuron", ("v", "m", "h", "n", "p", "s")),
    ("MedvedevMapNeuron", ("u",)),
    ("MerkelCell", ("v", "adapt")),
    ("MetaPlasticNeuron", ("v", "error_trace", "expected_reward")),
    ("MihalasNieburNeuron", ("v", "theta", "i1", "i2")),
    ("MontbrioMeanField", ("r", "v")),
    ("MorrisLecarNeuron", ("v", "w")),
    ("MotorUnit", ("v", "adapt", "force")),
    ("MultiTimescaleNeuron", ("v_fast", "v_medium", "v_slow")),
    ("MyelinatedAxon", ("v_inter",)),
    ("NeuroGridNeuron", ("v_s", "v_d")),
    ("Nociceptor", ("v", "sensitisation")),
    ("NodeOfRanvier", ("v", "m", "h", "p", "s")),
    ("NonlinearLIFNeuron", ("v", "w")),
    ("OlfactoryReceptorNeuron", ("v", "camp", "adapt", "pde4")),
    ("PVFastSpikingNeuron", ("v", "h", "n", "p")),
    ("PacinianCorpuscle", ("v", "prev_pressure", "adapt")),
    ("ParametricLIFNeuron", ("v",)),
    ("PernarowskiNeuron", ("v", "w", "z")),
    ("PlantR15Neuron", ("v", "m", "h", "n", "ca")),
    ("PospischilNeuron", ("v", "m", "h", "n", "p")),
    ("PredictiveCodingNeuron", ("v", "pred")),
    ("PrescottNeuron", ("v", "w")),
    ("QuadraticIFNeuron", ("v",)),
    ("RenshawCell", ("v", "h", "n", "adapt")),
    ("ResonateAndFireNeuron", ("x", "y")),
    ("RetinalGanglionCell", ("baseline", "on_centre")),
    ("RulkovMapNeuron", ("x", "y")),
    ("RustAdaptiveThresholdMoENeuron", ("v", "v_th")),
    ("SCClippedLogisticBurstingMapNeuron", ("x", "y")),
    ("SCClippedRationalRecoveryMapNeuron", ("x", "y")),
    ("SCFourStateGLIFNeuron", ("v", "theta", "i_asc1", "i_asc2")),
    ("SCResettingWilsonHRNeuron", ("v", "r")),
    ("SCScaledResetAdaptiveIFNeuron", ("v", "theta", "i1", "i2")),
    ("SCSixStateThalamocorticalNeuron", ("v", "h_na", "n_k", "m_h", "h_t", "na_i")),
    ("SCThreeStatePhantomBurster", ("v", "s1", "s2")),
    ("SCTriangularMcKeanNeuron", ("v", "w")),
    ("SCUpwardCrossingRulkovMapNeuron", ("x", "y")),
    ("SFANeuron", ("v", "g_sfa")),
    ("SSTNeuron", ("v", "m", "h", "n", "p", "s", "r")),
    ("SelfReferentialNeuron", ("v",)),
    ("ShermanRinzelKeizerNeuron", ("v", "n", "s")),
    ("SmoothMuscleCell", ("v", "ca", "ca_store")),
    ("SpiNNakerLIFNeuron", ("v", "refrac_count")),
    ("SpikeResponseNeuron", ("v", "time_since_spike")),
    ("StellateCell", ("v", "h", "n", "p")),
    ("TUMNetwork", ("r", "x", "u")),
    ("TermanWangOscillator", ("v", "w")),
    ("ThetaNeuron", ("theta",)),
    ("TraubMilesNeuron", ("v", "m", "h", "n")),
    ("UnipolarBrushCell", ("v", "persistent")),
    ("UpperMotorNeuron", ("v", "m", "h", "n", "p", "s")),
    ("VIPNeuron", ("v", "h", "n", "a", "b")),
    ("WangBuzsakiNeuron", ("v", "h", "n")),
    ("WilsonHRNeuron", ("v", "r")),
    ("YamadaNeuron", ("v", "n", "q")),
)


class _Neuron(Protocol):
    def step(self, current: object) -> int: ...

    def get_state(self) -> dict[str, float | bool]: ...

    def reset(self) -> object: ...


class _Constructor(Protocol):
    __name__: str
    __qualname__: str
    __module__: str

    def __call__(self, *args: object, **kwargs: object) -> _Neuron: ...


def _constructor(name: str) -> _Constructor:
    return cast(_Constructor, getattr(_NATIVE, name))


class DefaultNeuronBindingContracts(unittest.TestCase):
    """Check construction, scalar extraction, state ownership and serialization."""

    def test_source_producers_match_the_registered_state_contracts(self) -> None:
        """Reject an added or changed macro producer omitted by the runtime gate."""
        sources = Path(__file__).resolve().parents[1] / "engine/src/bindings"
        discovered: dict[str, tuple[str, ...]] = {}
        pattern = re.compile(r'py_neuron_default!\(\s*"([^"]+)"(.*?)\);', re.S)
        for source in sources.rglob("*.rs"):
            for name, arguments in pattern.findall(source.read_text(encoding="utf-8")):
                self.assertNotIn(name, discovered)
                discovered[name] = tuple(re.findall(r"state\s+(\w+)", arguments))
        self.assertEqual(discovered, dict(_DEFAULT_NEURONS))

    def test_native_global_class_identity_survives_every_pickle_protocol(self) -> None:
        """Resolve each exported class through its original native module."""
        for name, _ in _DEFAULT_NEURONS:
            constructor = _constructor(name)
            with self.subTest(model=name):
                self.assertEqual(constructor.__name__, name)
                self.assertEqual(constructor.__qualname__, name)
                self.assertEqual(constructor.__module__, _NAMESPACE)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(pickle.loads(pickle.dumps(constructor, protocol)), constructor)

    def test_constructor_refusals_preserve_the_named_zero_argument_api(self) -> None:
        """Refuse positional and keyword configuration at the actual constructor."""
        for name, _ in _DEFAULT_NEURONS:
            constructor = _constructor(name)
            for args, kwargs, message in [
                ((0.25,), {}, f"{name}.__new__() takes 0 positional arguments but 1 was given"),
                (
                    (0.25, 0.5),
                    {},
                    f"{name}.__new__() takes 0 positional arguments but 2 were given",
                ),
                (
                    (),
                    {"current": 0.25},
                    f"{name}.__new__() got an unexpected keyword argument 'current'",
                ),
                ((), {"v": -40.0}, f"{name}.__new__() got an unexpected keyword argument 'v'"),
            ]:
                with self.subTest(model=name, args=args, kwargs=kwargs):
                    with self.assertRaises(TypeError) as refusal:
                        constructor(*args, **kwargs)
                    self.assertEqual(str(refusal.exception), message)

    def test_returned_state_is_an_independent_mapping(self) -> None:
        """Prevent caller dictionary mutation from altering native state or another instance."""
        for name, fields in _DEFAULT_NEURONS:
            with self.subTest(model=name):
                first, other = _constructor(name)(), _constructor(name)()
                initial = first.get_state()
                self.assertEqual(tuple(initial), fields)
                self.assertEqual(other.get_state(), initial)
                returned = first.get_state()
                returned.clear()
                returned["external_key"] = 12345.0
                self.assertEqual(first.get_state(), initial)
                self.assertEqual(other.get_state(), initial)
                self.assertIsNot(first.get_state(), first.get_state())

    def test_failed_scalar_extraction_is_atomic_and_runtime_recovers(self) -> None:
        """Keep state unchanged on TypeError and match the next actual control step."""
        invalid_currents: tuple[object, ...] = (None, "text", [], (0.125,), 0.125j)
        for name, _ in _DEFAULT_NEURONS:
            for current in invalid_currents:
                with self.subTest(model=name, input_type=type(current).__name__):
                    neuron, control = _constructor(name)(), _constructor(name)()
                    before = neuron.get_state()
                    with self.assertRaises(TypeError) as refusal:
                        neuron.step(current)
                    self.assertEqual(
                        str(refusal.exception), f"must be real number, not {type(current).__name__}"
                    )
                    self.assertEqual(neuron.get_state(), before)
                    self.assertEqual(neuron.step(0.125), control.step(0.125))
                    self.assertEqual(neuron.get_state(), control.get_state())

    def test_numpy_scalar_and_readonly_zero_rank_inputs_follow_float_conversion(self) -> None:
        """Exercise real NumPy scalar extraction without changing caller storage."""
        values: list[tuple[object, float]] = [
            (np.float16(0.125), 0.125),
            (np.float32(0.125), 0.125),
            (np.float64(0.125), 0.125),
            (np.int64(1), 1.0),
            (np.bool_(True), 1.0),
        ]
        zero_rank = np.array(0.125)
        zero_rank.setflags(write=False)
        values.append((zero_rank, 0.125))
        for name, _ in _DEFAULT_NEURONS:
            for current, expected in values:
                with self.subTest(model=name, input_type=type(current).__name__):
                    neuron, control = _constructor(name)(), _constructor(name)()
                    before = np.asarray(current).tobytes()
                    self.assertEqual(neuron.step(current), control.step(expected))
                    self.assertEqual(neuron.get_state(), control.get_state())
                    self.assertEqual(np.asarray(current).tobytes(), before)
        self.assertFalse(zero_rank.flags.writeable)

    def test_instance_serialization_and_copy_refusals_preserve_native_state(self) -> None:
        """Refuse unsupported instance persistence for every advertised pickle protocol."""
        for name, _ in _DEFAULT_NEURONS:
            neuron = _constructor(name)()
            before = neuron.get_state()
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(model=name, protocol=protocol):
                    with self.assertRaises(TypeError) as refusal:
                        pickle.dumps(neuron, protocol)
                    label = name if protocol < 2 else f"{_NAMESPACE}.{name}"
                    self.assertEqual(str(refusal.exception), f"cannot pickle '{label}' object")
                    self.assertEqual(neuron.get_state(), before)
            for copier in [copy.copy, copy.deepcopy]:
                with self.subTest(model=name, copier=copier.__name__):
                    with self.assertRaises(TypeError) as refusal:
                        copier(neuron)
                    self.assertEqual(
                        str(refusal.exception), f"cannot pickle '{_NAMESPACE}.{name}' object"
                    )
                    self.assertEqual(neuron.get_state(), before)

    def test_temporal_state_and_reset_follow_the_declared_model_contract(self) -> None:
        """Retain Arcane deep memory while other default models reset to fresh state."""
        for name, fields in _DEFAULT_NEURONS:
            with self.subTest(model=name):
                neuron, control = _constructor(name)(), _constructor(name)()
                for current in [-0.1, 0.0, 0.25, 0.5, 1.0, 5.0] * 4:
                    output = neuron.step(current)
                    self.assertIs(type(output), int)
                    self.assertEqual(output, control.step(current))
                    self.assertEqual(neuron.get_state(), control.get_state())
                    self.assertEqual(tuple(neuron.get_state()), fields)
                before_reset = neuron.get_state()
                self.assertIsNone(neuron.reset())
                if name == "ArcaneNeuron":
                    self.assertEqual(neuron.get_state()["v_deep"], before_reset["v_deep"])
                    self.assertEqual(neuron.get_state()["v_fast"], 0.0)
                    self.assertEqual(neuron.get_state()["v_work"], 0.0)
                else:
                    fresh = _constructor(name)()
                    self.assertEqual(neuron.get_state(), fresh.get_state())
                    self.assertEqual(neuron.step(0.25), fresh.step(0.25))
                    self.assertEqual(neuron.get_state(), fresh.get_state())


if __name__ == "__main__":
    native_origin = Path(str(_NATIVE.__file__)).resolve()
    if not native_origin.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Default-neuron consumer requires an installed engine in this interpreter"
        )
    unittest.main()

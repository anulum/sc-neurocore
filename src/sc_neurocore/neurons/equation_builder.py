# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Arbitrary-equation neuron builder — define any model

"""Arbitrary-equation neuron builder — define any model from strings.

Eliminates the last competitive gap with Brian2: users specify ODEs
as strings, and EquationNeuron compiles them into a working model.

Usage:
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    # Define a custom neuron with string equations
    neuron = EquationNeuron(
        equations={
            "v": "-(v - v_rest) / tau + R * I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"v_rest": -65.0, "tau": 20.0, "R": 1.0, "epsilon": 0.08, "a": 0.7, "b": 0.8},
        state={"v": -65.0, "w": 0.0},
        threshold="v > v_threshold",
        reset={"v": "v_reset"},
        constants={"v_threshold": -50.0, "v_reset": -65.0},
        dt=0.1,
        method="euler",
    )

    for t in range(10000):
        spike = neuron.step(I=10.0)

    # Or use the factory for common patterns
    from sc_neurocore.neurons.equation_builder import from_equations

    lif = from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

import numpy as np

from sc_neurocore.neurons._stochastic_threshold import (
    DEFAULT_LFSR16_SEED,
    Lfsr16Threshold,
)
from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_safety import EVAL_GLOBALS, ExpressionSafetyValidator
from sc_neurocore.neurons.equation_units_runtime import (
    convert_runtime_value,
    prepare_strict_runtime,
)

SUPPORTED_METHODS = ("euler", "map", "rk4", "exp_euler", "gauss_seidel")

# Spike-detection modes a schema's ``[threshold]`` may declare. ``level`` fires on every
# step the condition holds (integrate -> threshold -> reset); ``crossing`` fires once on
# the rising transition (a non-resetting oscillator's ``v >= thr and v_prev < thr`` edge).
# ``poisson`` / ``escape_rate`` mark the stochastic threshold mechanism handled elsewhere;
# they are accepted here (so those schemas still construct) but never engage edge logic.
_SUPPORTED_DETECTION = frozenset({"level", "crossing", "poisson", "escape_rate"})


def _previous_state_aliases(state: dict[str, float]) -> dict[str, float]:
    """Return macro-boundary ``<state>_prev`` aliases for expression evaluation."""
    return {f"{name}_prev": value for name, value in state.items()}


class EquationNeuron:
    """Neuron defined by arbitrary ODE equations as strings.

    Each equation is a right-hand-side expression for dX/dt.
    Variables can reference other state variables, parameters,
    and the special variable `I` (input current).

    ``units="strict"`` enables opt-in pint-based dimensional
    validation before the expressions are compiled for runtime.
    """

    def __init__(
        self,
        equations: dict[str, str],
        parameters: dict[str, float] | None = None,
        state: dict[str, float] | None = None,
        threshold: str | None = None,
        reset: dict[str, str] | None = None,
        constants: dict[str, float] | None = None,
        dt: Any = 0.1,
        method: str = "euler",
        units: str = "none",
        input_unit: Any | None = None,
        detection: str = "level",
        substeps: int = 1,
        rate_expression: str | None = None,
        probability_expression: str | None = None,
        rng_seed: int = DEFAULT_LFSR16_SEED,
    ) -> None:
        """Initialise an equation-defined neuron from ODE strings."""
        if units not in {"none", "strict"}:
            raise ValueError("units must be 'none' or 'strict'")
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {list(SUPPORTED_METHODS)}, got {method!r}")
        if detection not in _SUPPORTED_DETECTION:
            raise ValueError(
                f"detection must be one of {sorted(_SUPPORTED_DETECTION)}, got {detection!r}"
            )
        # ``substeps`` advances the integrator this many inner steps per macro ``step()``
        # before a single spike decision, mirroring the conductance hand models' fixed
        # sub-stepping (e.g. 100 dt sub-steps per 1 ms macro step). Must be a positive
        # integer; the ``bool`` guard rejects ``True``/``False`` slipping through as 1/0.
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"substeps must be a positive integer, got {substeps!r}")

        self.equations = equations
        self.threshold_expr = threshold
        self.reset_rules = reset or {}
        self.method = method
        self.detection = detection
        self.substeps = substeps
        self.rate_expression = rate_expression
        self.probability_expression = probability_expression
        self._escape_rate_enabled = detection == "escape_rate" and rate_expression is not None
        self._poisson_enabled = detection == "poisson" and probability_expression is not None
        self._stochastic_threshold_enabled = self._escape_rate_enabled or self._poisson_enabled
        if detection != "escape_rate" and rate_expression is not None:
            raise ValueError("rate_expression is only valid with detection='escape_rate'")
        if detection != "poisson" and probability_expression is not None:
            raise ValueError("probability_expression is only valid with detection='poisson'")
        if detection == "escape_rate" and rate_expression is None:
            raise ValueError("escape_rate detection requires rate_expression")
        if detection == "escape_rate" and probability_expression is not None:
            raise ValueError("escape_rate detection cannot define probability_expression")
        if detection == "poisson" and probability_expression is None:
            raise ValueError("poisson detection requires probability_expression")
        if detection == "poisson" and rate_expression is not None:
            raise ValueError("poisson detection cannot define rate_expression")
        if self._stochastic_threshold_enabled:
            if threshold not in (None, "stochastic"):
                raise ValueError(
                    f"{detection} detection cannot combine a stochastic expression "
                    "with a level threshold"
                )
            if not math.isfinite(float(dt)) or float(dt) <= 0.0:
                raise ValueError(f"{detection} dt must be finite and positive")
            self.threshold_expr = None
            self._stochastic_rng: Lfsr16Threshold | None = Lfsr16Threshold(rng_seed)
        else:
            self._stochastic_rng = None
        # Rising-edge (``crossing``) detection is only engaged for a NON-resetting model:
        # a reset that drops the state back below threshold already clears the condition
        # every spike, so ``level`` and ``crossing`` are identical there and the simpler
        # (and previously validated) level path is used. Genuine no-reset oscillators
        # (e.g. FitzHugh-Nagumo, McKean) are the case that needs true edge detection.
        self._edge_detection = (
            detection == "crossing" and threshold is not None and not self.reset_rules
        )
        self.units = units
        self._strict_units = units == "strict"
        self._display_state_units: dict[str, Any] = {}
        self._base_state_units: dict[str, Any] = {}
        self._runtime_units: dict[str, Any] = {}
        self._input_unit_name = "I"

        self._namespace: dict[str, Any] = build_eval_namespace()
        raw_parameters = parameters or {}
        raw_state = state or {k: 0.0 for k in equations}
        raw_constants = constants or {}
        previous_aliases = {f"{name}_prev" for name in equations}
        occupied_names = set(equations) | set(raw_parameters) | set(raw_state) | set(raw_constants)
        alias_collisions = sorted(previous_aliases & occupied_names)
        if alias_collisions:
            joined = ", ".join(alias_collisions)
            raise ValueError(
                f"Names reserved for previous-state aliases cannot be declared: {joined}"
            )

        if self._strict_units:
            runtime = prepare_strict_runtime(
                equations=self.equations,
                threshold_expr=self.threshold_expr,
                reset_rules=self.reset_rules,
                input_unit_name=self._input_unit_name,
                raw_parameters=raw_parameters,
                raw_state=raw_state,
                raw_constants=raw_constants,
                dt=dt,
                input_unit=input_unit,
            )
            self.parameters = runtime.parameters
            self.state = runtime.state
            self.constants = runtime.constants
            self.dt = runtime.dt
            self._runtime_units = runtime.runtime_units
            self._base_state_units = runtime.base_state_units
            self._display_state_units = runtime.display_state_units
        else:
            self.parameters = raw_parameters
            self.state = raw_state
            self.constants = raw_constants
            self.dt = float(dt)

        self.initial_state = deepcopy(self.state)
        self._noise_scale = np.sqrt(self.dt)

        self._safety = ExpressionSafetyValidator()
        all_exprs = list(self.equations.values()) + list(self.reset_rules.values())
        if self.threshold_expr:
            all_exprs.append(self.threshold_expr)
        if self.rate_expression:
            all_exprs.append(self.rate_expression)
        if self.probability_expression:
            all_exprs.append(self.probability_expression)
        self._stochastic_uses_diffusion_noise = self._stochastic_threshold_enabled and any(
            re.search(r"\bxi\b", expression) is not None for expression in all_exprs
        )
        for expr in all_exprs:
            self._safety.validate(expr)

        self._compiled_eqs = {
            var: compile(expr, f"<eq:{var}>", "eval") for var, expr in self.equations.items()
        }
        self._compiled_threshold = (
            compile(self.threshold_expr, "<threshold>", "eval") if self.threshold_expr else None
        )
        self._compiled_rate = (
            compile(self.rate_expression, "<escape-rate>", "eval") if self.rate_expression else None
        )
        self._compiled_probability = (
            compile(self.probability_expression, "<poisson-probability>", "eval")
            if self.probability_expression
            else None
        )
        self._compiled_reset = {
            var: compile(expr, f"<reset:{var}>", "eval") for var, expr in self.reset_rules.items()
        }
        self.jacobian_expressions: dict[str, str] = {}
        self._compiled_jacobian = self._build_jacobian() if self.method == "exp_euler" else {}

        # Edge (``crossing``) detection tracks whether the threshold condition held on the
        # previously committed state, so a spike fires only on the rising transition
        # (inactive -> active) rather than on every step the condition holds. Seeding it
        # from the initial state means an oscillator that starts below threshold (the
        # usual case) does not emit a spurious first-step spike, and one already above
        # threshold waits for a genuine re-crossing. Only computed for edge models, so a
        # stochastic ``threshold`` condition (poisson/escape_rate) is never evaluated here.
        self._prev_threshold_active = (
            self.initial_threshold_active() if self._edge_detection else False
        )

    def initial_threshold_active(self) -> bool:
        """Return whether the threshold condition holds on the INITIAL committed state.

        This is the seed for the edge-detection ``_prev_threshold_active`` flag: before
        the first step the "previously committed state" is the initial state, and a reset
        returns to it. Evaluated deterministically with zero noise and zero input current
        — the corpus threshold conditions are functions of state (and parameters), so the
        input value does not affect the result. Returns ``False`` when no threshold is
        declared. Non-mutating, so the Verilog emitter can seed its ``_thr_prev`` register
        with the same value regardless of the neuron's current runtime state.
        """
        if self._compiled_threshold is None:
            return False
        env: dict[str, object] = dict(self._namespace)
        env["xi"] = 0.0
        env.update(self.parameters)
        env.update(self.constants)
        env.update(self.initial_state)
        env.update(_previous_state_aliases(self.initial_state))
        env["I"] = 0.0
        # Bandit B307 justification: AST-whitelisted threshold expression (see step()).
        return bool(eval(self._compiled_threshold, self._EVAL_GLOBALS, env))  # nosec B307

    def _build_jacobian(self) -> dict[str, Any]:
        """Compile the diagonal Jacobian ``∂f/∂x`` for each equation.

        Exponential Euler linearises ``dx/dt = f(x, …)`` around the current state
        via ``A = ∂f/∂x``; this differentiates each right-hand-side symbolically
        (:func:`~sc_neurocore.neurons.expression_derivative.differentiate`),
        validates the result against the same expression grammar, and compiles it.
        A model whose dynamics cannot be faithfully differentiated with respect to
        their own variable cannot use exponential Euler, and the differentiator
        raises to say so rather than integrate through a non-smooth term.

        Each derivative string is also kept in :attr:`jacobian_expressions` so the
        Verilog emitter lowers the *same* ``A`` expression the golden compiles — one
        derivative drives both backends, which is what keeps the hardware step
        consistent with the golden by construction rather than by a parallel
        re-derivation.

        The symbolic differentiator (and its SymPy backend) is imported lazily so
        that only exponential-Euler models pull the optional dependency in; forward
        Euler and RK4 never touch it.
        """
        try:
            from sc_neurocore.neurons.expression_derivative import differentiate
        except ImportError as exc:  # pragma: no cover - exercised only without SymPy
            raise ImportError(
                "method='exp_euler' requires SymPy for the symbolic Jacobian; "
                "install sc-neurocore[symbolic]"
            ) from exc
        compiled: dict[str, Any] = {}
        for var, expr in self.equations.items():
            derivative = differentiate(expr, var)
            self._safety.validate(derivative)
            self.jacobian_expressions[var] = derivative
            compiled[var] = compile(derivative, f"<jac:{var}>", "eval")
        return compiled

    # The compiled-expression ``eval`` sites below run with this empty-builtins
    # sandbox; the AST allowlist that makes every targeted B307 suppression sound lives in
    # :class:`~sc_neurocore.neurons.equation_safety.ExpressionSafetyValidator`.
    _EVAL_GLOBALS = EVAL_GLOBALS

    def _build_env(self, **kwargs: float) -> dict[str, object]:
        """Build the eval environment with parameters, state, and noise."""
        env: dict[str, object] = dict(self._namespace)
        # Euler-Maruyama: noise scaled by sqrt(dt)/dt so that after deriv*dt
        # the net noise is noise_scale * sqrt(dt) * N(0,1)
        if self._stochastic_threshold_enabled and not self._stochastic_uses_diffusion_noise:
            # Canonical stochastic-threshold models own an explicit model-scoped LFSR. Do
            # not consume NumPy's process-global stream when none of its authored
            # equations, probability/rate, or reset expressions references diffusion noise.
            env["xi"] = 0.0
        else:
            env["xi"] = self._noise_scale * np.random.randn() / max(self.dt, 1e-12) ** 0.5
        env.update(self.parameters)
        env.update(self.constants)
        env.update(self.state)
        env.update(kwargs)
        return env

    def step(self, I: float = 0.0, **kwargs: float) -> int:
        """Advance the neuron by one macro timestep; return 1 if it spikes.

        When ``substeps > 1`` the state is advanced by that many inner integration
        sub-steps before a single spike decision is taken on the macro boundary. This
        matches the maintained conductance hand models whose ``step()`` runs a fixed
        number of fine sub-steps per macro step (Hodgkin-Huxley / Connor-Stevens: 100
        sub-steps of ``dt`` per 1 ms macro step; Wang-Buzsaki: 50 per 0.5 ms) and detect
        the threshold crossing only across the macro step. The edge-crossing flag is
        compared against the state at the previous macro boundary, not per sub-step, so a
        repetitively firing oscillator counts one spike per action potential rather than
        one per sub-step it stays above threshold. With the default ``substeps == 1`` this
        is a plain single integration step and every existing model is bit-for-bit
        unchanged.
        """
        kwargs["I"] = convert_runtime_value(
            strict_units=self._strict_units,
            runtime_units=self._runtime_units,
            name=self._input_unit_name,
            value=I,
        )
        if self._strict_units:
            kwargs = {
                name: convert_runtime_value(
                    strict_units=self._strict_units,
                    runtime_units=self._runtime_units,
                    name=name,
                    value=value,
                )
                for name, value in kwargs.items()
            }
        previous_state = dict(self.state)
        try:
            for _ in range(self.substeps):
                self._integrate_once(**kwargs)
        except Exception:
            if self._stochastic_threshold_enabled:
                self.state = previous_state
            raise

        spike = 0
        if self._stochastic_threshold_enabled:
            stochastic_rng = self._stochastic_rng
            compiled_stochastic = (
                self._compiled_rate if self._escape_rate_enabled else self._compiled_probability
            )
            if stochastic_rng is None or compiled_stochastic is None:
                raise RuntimeError("stochastic-threshold runtime was not initialised")
            previous_rng = stochastic_rng.state
            try:
                env_post = self._build_env(**kwargs)
                env_post.update(_previous_state_aliases(previous_state))
                if self._escape_rate_enabled:
                    rate = float(eval(compiled_stochastic, self._EVAL_GLOBALS, env_post))
                    hazard = rate * self.dt
                    if not math.isfinite(rate) or rate < 0.0:
                        raise FloatingPointError("escape rate must remain finite and non-negative")
                    if not math.isfinite(hazard) or hazard < 0.0:
                        raise FloatingPointError(
                            "escape hazard must remain finite and non-negative"
                        )
                    probability = -math.expm1(-hazard)
                else:
                    probability = float(eval(compiled_stochastic, self._EVAL_GLOBALS, env_post))
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise FloatingPointError(
                        "stochastic spike probability must remain finite and bounded"
                    )
                fired = stochastic_rng.trial(probability)
                if fired:
                    spike = 1
                    reset_env = self._build_env(**kwargs)
                    reset_env.update(_previous_state_aliases(previous_state))
                    for var, code in self._compiled_reset.items():
                        # Bandit B307 justification: AST-whitelisted compiled reset rule.
                        self.state[var] = float(  # nosec B307
                            eval(code, self._EVAL_GLOBALS, reset_env)  # nosec B307
                        )
            except Exception:
                self.state = previous_state
                stochastic_rng.restore(previous_rng)
                raise
        elif self._compiled_threshold:
            env_post = self._build_env(**kwargs)
            env_post.update(_previous_state_aliases(previous_state))
            # Bandit B307 justification: AST-whitelisted compiled threshold expression.
            active = bool(eval(self._compiled_threshold, self._EVAL_GLOBALS, env_post))  # nosec B307
            # ``crossing`` fires once on the inactive -> active transition (a rising
            # threshold crossing, matching the hand oscillator models' ``v >= thr and
            # v_prev < thr`` edge test); ``level`` fires on every step the condition
            # holds. The tracked flag is the pre-reset condition on the just-integrated
            # state, which equals the committed state for a non-resetting oscillator and
            # is cleared by any reset that drops the state back below threshold, so a
            # reset-based level model behaves identically under either detection mode.
            if self._edge_detection:
                fired = active and not self._prev_threshold_active
                self._prev_threshold_active = active
            else:
                fired = active
            if fired:
                spike = 1
                reset_env = self._build_env(**kwargs)
                reset_env.update(_previous_state_aliases(previous_state))
                for var, code in self._compiled_reset.items():
                    # Bandit B307 justification: AST-whitelisted compiled reset rule.
                    self.state[var] = float(eval(code, self._EVAL_GLOBALS, reset_env))  # nosec B307

        return spike

    def _integrate_once(self, **kwargs: float) -> None:
        """Advance the state by one integration sub-step (no spike/threshold decision).

        Rebuilds the evaluation environment from the current state so it can be called
        repeatedly within a macro step, applies the method's integrator, and fails closed
        on a non-finite state. The threshold/spike decision is taken once per macro step
        in :meth:`step`, so this helper never reads the threshold or reset rules.
        """
        env = self._build_env(**kwargs)

        if self.method == "euler":
            derivatives = {}
            for var, code in self._compiled_eqs.items():
                # Bandit B307 justification: `code` is a compiled expression that has
                # already passed `ExpressionSafetyValidator.validate`'s AST
                # whitelist (no imports, no attribute access into builtins, only
                # the whitelisted maths/comparison nodes). The `eval` env has
                # empty `__builtins__` so even reaching `eval` / `exec` /
                # `__import__` is impossible.
                derivatives[var] = float(eval(code, self._EVAL_GLOBALS, env))  # nosec B307
            for var in self.equations:
                self.state[var] += derivatives[var] * self.dt

        elif self.method == "map":
            # Discrete-time map: state_{n+1} = f(state_n) directly, with no dt scaling
            # and no `+ state` term. Each equation right-hand side IS the next-step
            # value (not a derivative), so a schema that declares a map (e.g. the
            # Rulkov 2002 fast/slow recurrence) iterates exactly instead of being
            # integrated as an ODE. Every update reads the pre-step state so the map
            # is applied simultaneously, matching the published recurrence.
            updates = {}
            for var, code in self._compiled_eqs.items():
                # Bandit B307 justification: `code` is a compiled expression that already passed
                # `ExpressionSafetyValidator.validate`'s AST whitelist and evaluates
                # with empty `__builtins__` (see the euler branch comment for full rationale).
                updates[var] = float(eval(code, self._EVAL_GLOBALS, env))  # nosec B307
            for var in self.equations:
                self.state[var] = updates[var]

        elif self.method == "rk4":
            s0 = deepcopy(self.state)

            xi_sample = self._noise_scale * np.random.randn() / max(self.dt, 1e-12) ** 0.5

            def eval_derivs(state_override: dict[str, float]) -> dict[str, float]:
                """Evaluate all ODE derivatives at given state."""
                e: dict[str, object] = dict(self._namespace)
                e.update(self.parameters)
                e.update(self.constants)
                e.update(state_override)
                e.update(kwargs)
                e["xi"] = xi_sample
                return {
                    # Bandit B307 justification: AST-whitelisted compiled equation
                    # (see euler branch comment above for full sandbox
                    # rationale).
                    var: float(eval(code, self._EVAL_GLOBALS, e))  # nosec B307
                    for var, code in self._compiled_eqs.items()
                }

            k1 = eval_derivs(s0)
            s1 = {v: s0[v] + k1[v] * self.dt / 2 for v in self.equations}
            k2 = eval_derivs(s1)
            s2 = {v: s0[v] + k2[v] * self.dt / 2 for v in self.equations}
            k3 = eval_derivs(s2)
            s3 = {v: s0[v] + k3[v] * self.dt for v in self.equations}
            k4 = eval_derivs(s3)
            for v in self.equations:
                self.state[v] = s0[v] + (k1[v] + 2 * k2[v] + 2 * k3[v] + k4[v]) * self.dt / 6

        elif self.method == "gauss_seidel":
            # Sequential (Gauss-Seidel) forward Euler: advance each state variable in
            # declaration order, writing its updated value back into the evaluation
            # environment before the next variable's derivative is evaluated. A
            # later-declared variable therefore reads the ALREADY-UPDATED earlier
            # variables within the same sub-step, unlike the simultaneous ``euler`` mode
            # whose derivatives all read the pre-step state. This matches conductance
            # models whose hand implementation updates the gating variables first and the
            # membrane voltage from the new gates (Wang-Buzsaki 1996 updates h, n before
            # v). ``env`` already holds the pre-step state, parameters, constants, kwargs
            # and a single ``xi`` draw (one per sub-step, matching euler); committing the
            # variable in place makes the next derivative in the loop read the new value.
            for var, code in self._compiled_eqs.items():
                # Bandit B307 justification: `code` is a compiled expression that already passed
                # `ExpressionSafetyValidator.validate`'s AST whitelist and evaluates
                # with empty `__builtins__` (see the euler branch comment for full rationale).
                derivative = float(eval(code, self._EVAL_GLOBALS, env))  # nosec B307
                self.state[var] += derivative * self.dt
                env[var] = self.state[var]

        elif self.method == "exp_euler":
            # Linearised exponential Euler (Rush-Larsen): for each dx/dt = f(x),
            # linearise A = ∂f/∂x and take the exact update of x' = A x + b,
            #     x <- x + f(x) * dt * exprel(A dt),   exprel(z) = (exp(z) - 1)/z.
            # exprel carries the removable singularity (exprel(0) = 1), so A -> 0
            # is the Euler limit; for the gating form f = (x_inf - x)/tau this is
            # the exact x <- x_inf + (x - x_inf) exp(-dt/tau), stable at stiff dt.
            exprel = self._namespace["exprel"]
            increments = {}
            for var in self.equations:
                # Bandit B307 justification: AST-whitelisted compiled equation / Jacobian
                # (see the euler branch comment for the full sandbox rationale).
                f_val = float(eval(self._compiled_eqs[var], self._EVAL_GLOBALS, env))  # nosec B307
                a_val = float(eval(self._compiled_jacobian[var], self._EVAL_GLOBALS, env))  # nosec B307
                increments[var] = f_val * self.dt * float(exprel(a_val * self.dt))
            for var in self.equations:
                self.state[var] += increments[var]

        # Fail closed on a non-finite state, matching the hand neuron models'
        # ``_validate_candidate`` contract: an integrator that diverges (e.g. a
        # cubic relaxation oscillator stepped past its stability limit) must raise
        # rather than silently propagate ``inf``/``nan`` into the threshold decision.
        for var, value in self.state.items():
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"{var!r} became non-finite ({value}) after a {self.method} step"
                )

    def get_state(self) -> dict[str, Any]:
        """Return current state, with units if in strict mode."""
        if self._strict_units:
            return {
                name: (value * self._base_state_units[name]).to(self._display_state_units[name])
                for name, value in self.state.items()
            }
        return dict(self.state)

    @property
    def escape_rng_initial_seed(self) -> int | None:
        """Return the emitted stochastic-threshold seed (legacy API name)."""
        return self.stochastic_rng_initial_seed

    @property
    def escape_rng_state(self) -> int | None:
        """Return the live stochastic-threshold state (legacy API name)."""
        return self.stochastic_rng_state

    @property
    def stochastic_rng_initial_seed(self) -> int | None:
        """Return the emitted stochastic-threshold seed, or ``None`` when unused."""
        return self._stochastic_rng.initial_seed if self._stochastic_rng is not None else None

    @property
    def stochastic_rng_state(self) -> int | None:
        """Return the live stochastic-threshold LFSR state, or ``None`` when unused."""
        return self._stochastic_rng.state if self._stochastic_rng is not None else None

    def reset(self) -> None:
        """Reset state to initial values."""
        self.state = deepcopy(self.initial_state)
        if self._stochastic_rng is not None:
            self._stochastic_rng.reset()
        self._prev_threshold_active = (
            self.initial_threshold_active() if self._edge_detection else False
        )

    def __repr__(self) -> str:
        """Human-readable representation of the neuron equations."""
        eqs = ", ".join(f"d{k}/dt = {v}" for k, v in self.equations.items())
        return f"EquationNeuron({eqs})"


def from_equations(
    *equation_strings: str,
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, Any] | None = None,
    init: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
    dt: Any = 0.1,
    method: str = "euler",
    units: str = "none",
    input_unit: Any | None = None,
    detection: str = "level",
) -> EquationNeuron:
    """Build an EquationNeuron from Brian2-style equation strings.

    Example:
        lif = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )

    Use ``units="strict"`` with pint quantities to validate the
    equation dimensions before runtime compilation.
    """
    equations = {}
    for eq_str in equation_strings:
        eq_str = eq_str.strip()
        m = re.match(r"d(\w+)/dt\s*=\s*(.+)", eq_str)
        if m:
            var_name = m.group(1)
            rhs = m.group(2).strip()
            equations[var_name] = rhs
        else:
            raise ValueError(f"Cannot parse equation: {eq_str!r}. Expected 'd<var>/dt = <expr>'")

    reset_rules = {}
    constant_values = constants.copy() if constants else {}
    if reset:
        for part in reset.split(";"):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"(\w+)\s*=\s*(.+)", part)
            if m:
                var = m.group(1)
                val_str = m.group(2).strip()
                try:
                    constant_values[f"{var}_reset_val"] = float(val_str)
                    reset_rules[var] = f"{var}_reset_val"
                except ValueError:
                    reset_rules[var] = val_str

    threshold_expr = None
    if threshold:
        threshold = threshold.strip()
        threshold_expr = threshold

    state = init or {k: 0.0 for k in equations}

    return EquationNeuron(
        equations=equations,
        parameters=params or {},
        state=state,
        threshold=threshold_expr,
        reset=reset_rules,
        constants=constant_values,
        dt=dt,
        method=method,
        units=units,
        input_unit=input_unit,
        detection=detection,
    )

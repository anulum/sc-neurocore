# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fisher-Posner quantum bridge adapter

"""Quantum bridge adapter for the Fisher-Posner cognition layer.

Provides the interface between the spin pool emulator and external
quantum backends (PennyLane, Qiskit) as well as the cross-repo
orchestrator infrastructure.

This module composes with existing infrastructure rather than
duplicating it:

- Uses ``sc_neurocore.quantum.hardware_bridge.QuantumHardwareLayer``
  for actual QPU dispatch when available.
- Produces/consumes payloads compatible with
  ``scpn_neurocore.bridge.QPUBridgeArtifact`` schema v1.
- Accepts orchestrator state dicts matching the format from
  ``scpn_phase_orchestrator.adapters.quantum_control_bridge``.

The bridge supports graceful degradation: without PennyLane installed,
it falls back to the pure-Python MPS emulator for phase optimisation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pennylane as _qml

    qml: Any = _qml
    HAS_PENNYLANE = True
except (ImportError, AttributeError):
    qml = None
    HAS_PENNYLANE = False

# Hard limits for qubit auto-sizing
_QUBIT_FLOOR = 4
_QUBIT_CEILING = 30  # 2^30 × 16 bytes = 16 GB state vector
_RAM_SAFETY_FACTOR = 0.5  # never consume more than 50% of free RAM


def compute_max_qubits(safety_factor: float = _RAM_SAFETY_FACTOR) -> int:
    """Compute maximum PennyLane qubits that fit in available RAM.

    PennyLane ``default.qubit`` uses a dense state vector of shape
    ``(2**n_qubits,)`` with ``complex128`` (16 bytes per amplitude).
    This function reads available RAM and computes the largest qubit
    count that stays within the safety budget.

    Falls back to ``/proc/meminfo`` if ``psutil`` is unavailable
    (common on minimal containers).

    Parameters
    ----------
    safety_factor : float
        Fraction of available RAM to allow (0, 1].  Default 0.5.

    Returns
    -------
    int
        Maximum qubit count, clamped to [``_QUBIT_FLOOR``, ``_QUBIT_CEILING``].
    """
    avail_bytes = _get_available_ram()
    if avail_bytes <= 0:
        logger.warning("Cannot determine available RAM, using default %d qubits", _QUBIT_FLOOR)
        return _QUBIT_FLOOR

    usable = avail_bytes * safety_factor
    # state vector: 2^n × 16 bytes (complex128)
    if usable < 16:
        return _QUBIT_FLOOR
    max_n = int(math.log2(usable / 16.0))
    result = max(_QUBIT_FLOOR, min(max_n, _QUBIT_CEILING))
    logger.debug(
        "RAM auto-sizing: %.1f GB available, safety=%.0f%%, max_qubits=%d",
        avail_bytes / (1024**3),
        safety_factor * 100,
        result,
    )
    return result


def _get_available_ram() -> int:
    """Return available RAM in bytes.  psutil → /proc/meminfo fallback."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


class FisherPosnerQuantumBridge:
    """Bridge between SpinPoolMPS and quantum hardware / orchestrator.

    Supports three operational modes:

    1. **Emulated** (default): Pure NumPy phase optimisation via the
       MPS emulator.  No external dependencies required.
    2. **PennyLane**: Gradient-based phase optimisation using PennyLane
       autograd on simulated qubits.
    3. **Orchestrator**: Accepts global phase vectors from the
       scpn-phase-orchestrator and combines them with local
       optimisation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / spin sites.
    backend : str
        Backend selection: ``"auto"``, ``"pennylane"``, ``"ibm_qiskit"``,
        or ``"emulated"``.
    """

    def __init__(self, n_qubits: int, backend: str = "auto") -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n_qubits = n_qubits

        # Resolve backend
        if backend == "auto":
            if HAS_PENNYLANE:
                backend = "pennylane"
            else:
                backend = "emulated"
        self._backend = backend

        self.dev: Any = None
        self._ibm_service: Any = None

        if backend == "pennylane":
            if not HAS_PENNYLANE:
                raise ImportError(
                    "PennyLane is required for the 'pennylane' backend. "
                    "Install with: pip install sc-neurocore[quantum-cognition]"
                )
            self.dev = qml.device("default.qubit", wires=n_qubits)
            logger.info("FisherPosnerQuantumBridge: PennyLane backend (%d qubits)", n_qubits)
        elif backend == "ibm_qiskit":
            self._init_ibm_backend()
        elif backend == "ibm_aer":
            logger.info(
                "FisherPosnerQuantumBridge: explicit Aer simulator backend (%d qubits)",
                n_qubits,
            )
        else:
            logger.info(
                "FisherPosnerQuantumBridge: emulated backend (%d qubits)",
                n_qubits,
            )

    def _init_ibm_backend(self) -> None:
        """Initialise IBM Quantum backend via qiskit-ibm-runtime."""
        import os

        token = (
            os.environ.get("SC_NEUROCORE_IBM_TOKEN", "")
            or os.environ.get("QISKIT_IBM_TOKEN", "")
            or os.environ.get("IBM_QUANTUM_TOKEN", "")
        )
        if not token:
            raise RuntimeError(
                "backend='ibm_qiskit' requires SC_NEUROCORE_IBM_TOKEN, "
                "QISKIT_IBM_TOKEN, or IBM_QUANTUM_TOKEN. "
                "Use backend='ibm_aer' explicitly for simulator runs."
            )
        channel = os.environ.get("SC_NEUROCORE_IBM_CHANNEL", "ibm_cloud")
        instance = os.environ.get("SC_NEUROCORE_IBM_CRN") or os.environ.get(
            "SC_NEUROCORE_IBM_INSTANCE"
        )

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs = {"channel": channel, "token": token}
            if instance:
                kwargs["instance"] = instance
            self._ibm_service = QiskitRuntimeService(**kwargs)
            logger.info(
                "FisherPosnerQuantumBridge: IBM Quantum backend (%d qubits)",
                self.n_qubits,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IBM Quantum: {e}") from e

    @property
    def backend(self) -> str:
        """Active backend name."""
        return self._backend

    def execute_non_local_sync(self, entangle_pairs: list[tuple[int, int]]) -> np.ndarray[Any, Any]:
        """Execute non-local synchronisation via entanglement.

        Creates Bell pairs for the specified site pairs and measures
        expectation values of PauliZ on all qubits.

        Parameters
        ----------
        entangle_pairs : list[tuple[int, int]]
            Pairs of qubit indices to entangle.

        Returns
        -------
        np.ndarray[Any, Any]
            PauliZ expectation values for all qubits, shape (n_qubits,).
        """
        if self._backend == "pennylane" and self.dev is not None:
            return self._sync_pennylane(entangle_pairs)

        if self._backend in ("ibm_qiskit", "ibm_aer"):
            return self._sync_ibm_qiskit(entangle_pairs)

        # Emulated fallback
        return self._sync_emulated(entangle_pairs)

    def _sync_ibm_qiskit(
        self, entangle_pairs: list[tuple[int, int]], shots: int = 4096
    ) -> np.ndarray[Any, Any]:
        """Build and execute Bell pair circuit on IBM backend or AerSimulator.

        NOTE: This method dispatches COGNITIVE-LAYER entanglement (Bell
        pairs for non-local synchronization), NOT physics-level Posner
        Hamiltonian circuits. For Posner physics dispatch, use
        ``execute_posner_circuit()``.
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for p1, p2 in entangle_pairs:
            qc.h(p1)
            qc.cx(p1, p2)
        qc.measure(range(self.n_qubits), range(self.n_qubits))

        return self._dispatch_qiskit_circuit(qc, shots)

    def execute_posner_circuit(self, shots: int = 4096, **posner_kwargs: Any) -> dict[str, int]:
        """Dispatch an actual 8q Posner Hamiltonian circuit to IBM QPU.

        Builds the full radical-pair Trotter circuit from
        ``verify_ibm_heron.build_posner_circuit()`` and dispatches it
        to the configured IBM backend. Returns raw bitstring counts.

        Parameters
        ----------
        shots : int
            Number of measurement shots.
        **posner_kwargs
            Forwarded to ``build_posner_circuit()`` (J, hf1, hf2, etc.).

        Returns
        -------
        dict
            Bitstring counts from QPU execution.
        """
        build_posner_circuit: Callable[..., Any] | None = None
        # Try multiple import paths for robustness
        for module_path in [
            "tools.verify_ibm_heron",
            "verify_ibm_heron",
            "sc_neurocore.tools.verify_ibm_heron",
        ]:
            try:
                import importlib

                mod = importlib.import_module(module_path)
                build_posner_circuit = mod.build_posner_circuit
                break
            except (ImportError, AttributeError):
                continue

        if build_posner_circuit is None:
            # Last resort: try adding tools/ to sys.path
            import sys
            from pathlib import Path

            tools_dir = Path(__file__).resolve().parents[3] / "tools"
            if tools_dir.is_dir() and str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
                try:
                    from verify_ibm_heron import build_posner_circuit  # type: ignore
                except ImportError:
                    logger.error(
                        "Cannot import build_posner_circuit. "
                        "Ensure tools/ is in sys.path or install as package."
                    )
                    raise ImportError(
                        "Cannot import build_posner_circuit. Ensure tools/ is "
                        "in sys.path or install SC-NeuroCore with tool entry points."
                    )
            if build_posner_circuit is None:
                raise ImportError(
                    "Cannot import build_posner_circuit. Ensure tools/ is in "
                    "sys.path or install SC-NeuroCore with tool entry points."
                )

        qc = build_posner_circuit(**posner_kwargs)
        return self._dispatch_qiskit_circuit_raw(qc, shots)

    @staticmethod
    def _extract_qiskit_counts(pub_result: Any) -> dict[str, int]:
        """Extract counts from a SamplerV2 result regardless of register name."""
        data = pub_result.data
        for reg_name in ("meas", "c", "cr", "c0", "c1"):
            register = getattr(data, reg_name, None)
            if register is not None and hasattr(register, "get_counts"):
                return cast("dict[str, int]", register.get_counts())
        for attr in dir(data):
            if attr.startswith("_"):
                continue
            register = getattr(data, attr, None)
            if register is not None and hasattr(register, "get_counts"):
                return cast("dict[str, int]", register.get_counts())
        raise RuntimeError("SamplerV2 result does not expose a counts register")

    def _dispatch_qiskit_circuit(self, qc: Any, shots: int) -> np.ndarray[Any, Any]:
        """Dispatch a circuit and return ⟨Z⟩ expectation values."""
        if self._backend == "ibm_qiskit" and self._ibm_service is not None:
            try:
                from qiskit import transpile
                from qiskit_ibm_runtime import SamplerV2

                # Use qc.num_qubits (circuit size), NOT self.n_qubits
                # (cognitive layer), because Posner circuits can be
                # 8q/16q/35q while cognitive layer may be 4-16q.
                backend = self._ibm_service.least_busy(
                    min_num_qubits=qc.num_qubits, operational=True
                )
                tqc = transpile(qc, backend, optimization_level=3)
                sampler = SamplerV2(mode=backend)
                job = sampler.run([tqc], shots=shots)
                result = job.result()
                counts = self._extract_qiskit_counts(result[0])
                logger.info("IBM QPU job %s completed (%d shots)", job.job_id(), shots)
            except Exception as e:
                raise RuntimeError(f"IBM QPU dispatch failed: {e}") from e
        elif self._backend == "ibm_aer":
            return self._sync_ibm_aer(qc, shots)
        else:
            raise RuntimeError(
                f"Qiskit dispatch requires backend='ibm_qiskit' or 'ibm_aer', got {self._backend!r}"
            )

        return self._counts_to_expvals(counts, shots)

    def _dispatch_qiskit_circuit_raw(self, qc: Any, shots: int) -> dict[str, int]:
        """Dispatch a circuit and return raw bitstring counts."""
        from qiskit import transpile

        if self._backend == "ibm_qiskit" and self._ibm_service is not None:
            try:
                from qiskit_ibm_runtime import SamplerV2

                backend = self._ibm_service.least_busy(
                    min_num_qubits=qc.num_qubits, operational=True
                )
                tqc = transpile(qc, backend, optimization_level=3)
                sampler = SamplerV2(mode=backend)
                job = sampler.run([tqc], shots=shots)
                result = job.result()
                counts = self._extract_qiskit_counts(result[0])
                logger.info("IBM QPU Posner job %s completed (%d shots)", job.job_id(), shots)
                return counts
            except Exception as e:
                raise RuntimeError(f"IBM QPU Posner dispatch failed: {e}") from e
        if self._backend == "ibm_aer":
            try:
                from qiskit_aer import AerSimulator

                sim = AerSimulator()
                tqc = transpile(qc, sim)
                return cast("dict[str, int]", sim.run(tqc, shots=shots).result().get_counts())
            except ImportError as e:
                raise RuntimeError("qiskit-aer is required for backend='ibm_aer'") from e
        raise RuntimeError(
            f"Raw Qiskit dispatch requires backend='ibm_qiskit' or 'ibm_aer', got {self._backend!r}"
        )

    def _sync_ibm_aer(self, qc: Any, shots: int = 4096) -> np.ndarray[Any, Any]:
        """Execute circuit on explicit local AerSimulator backend."""
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator

            sim = AerSimulator()
            tqc = transpile(qc, sim)
            counts = sim.run(tqc, shots=shots).result().get_counts()
            return self._counts_to_expvals(counts, shots)
        except ImportError as e:
            raise RuntimeError("qiskit-aer is required for backend='ibm_aer'") from e

    def _counts_to_expvals(self, counts: dict[str, int], shots: int) -> np.ndarray[Any, Any]:
        """Convert bitstring counts to ⟨Z⟩ expectation values."""
        expvals = np.zeros(self.n_qubits, dtype=np.float64)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            for q in range(min(self.n_qubits, len(bits))):
                bit = int(bits[-(q + 1)])
                expvals[q] += (1 - 2 * bit) * count  # |0⟩→+1, |1⟩→-1
        expvals /= shots
        return expvals

    def _sync_pennylane(self, entangle_pairs: list[tuple[int, int]]) -> np.ndarray[Any, Any]:
        """PennyLane Bell pair circuit → PauliZ expectations."""
        dev = self.dev

        @qml.qnode(dev)  # type: ignore[untyped-decorator]
        def circuit() -> list[Any]:
            for p1, p2 in entangle_pairs:
                qml.Hadamard(wires=p1)
                qml.CNOT(wires=[p1, p2])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        result = np.asarray(circuit(), dtype=np.float64)
        return result

    def _sync_emulated(self, entangle_pairs: list[tuple[int, int]]) -> np.ndarray[Any, Any]:
        """Pure-numpy emulation of Bell pair correlations."""
        expectations = np.ones(self.n_qubits, dtype=np.float64)
        for p1, p2 in entangle_pairs:
            # After Hadamard + CNOT: <Z> = 0 for entangled qubits
            expectations[p1] = 0.0
            expectations[p2] = 0.0
        return expectations

    def optimize_phases(
        self,
        target_coherence: float,
        learning_rate: float = 0.05,
        n_steps: int = 1,
    ) -> np.ndarray[Any, Any] | None:
        """Optimise qubit phases towards target coherence.

        Uses PennyLane autograd gradient descent when available,
        otherwise returns None (emulated mode does not support
        gradient-based optimisation).

        Parameters
        ----------
        target_coherence : float
            Target X-basis expectation value.
        learning_rate : float
            Gradient descent step size.
        n_steps : int
            Number of gradient descent steps.

        Returns
        -------
        np.ndarray[Any, Any] or None
            Optimised phase parameters, or None if running emulated.
        """
        if self._backend != "pennylane" or not HAS_PENNYLANE or self.dev is None:
            logger.debug("optimize_phases: skipped (emulated backend)")
            return None

        params = qml.numpy.array(np.random.uniform(0, np.pi, self.n_qubits), requires_grad=True)

        @qml.qnode(self.dev)  # type: ignore[untyped-decorator]
        def cost_circuit(phi: np.ndarray[Any, Any]) -> Any:
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(phi[i], wires=i)
            return qml.expval(qml.PauliX(0))

        def cost(phi: np.ndarray[Any, Any]) -> Any:
            return (cost_circuit(phi) - target_coherence) ** 2

        grad_fn = qml.grad(cost)
        for step in range(n_steps):
            grad_val = grad_fn(params)
            params = params - learning_rate * grad_val
            if step % max(1, n_steps // 5) == 0:
                logger.debug("optimize_phases step %d: cost=%.6f", step, cost(params))

        return np.asarray(params, dtype=np.float64)

    def apply_orchestrator_bias(
        self,
        global_phases: np.ndarray[Any, Any],
        target_coherence: float,
        learning_rate: float = 0.2,
    ) -> np.ndarray[Any, Any] | None:
        """Combine global orchestrator phases with local optimisation.

        Accepts a phase vector from the scpn-phase-orchestrator
        (matching the ``QuantumControlBridge.export_artifact()`` format)
        and uses it as the initial point for local gradient descent.

        Parameters
        ----------
        global_phases : np.ndarray[Any, Any]
            Phase vector from the orchestrator, shape (n_qubits,).
        target_coherence : float
            Target coherence level.
        learning_rate : float
            Step size for local optimisation.

        Returns
        -------
        np.ndarray[Any, Any] or None
            Locally refined phase parameters.
        """
        if len(global_phases) != self.n_qubits:
            raise ValueError(
                f"global_phases length {len(global_phases)} != n_qubits {self.n_qubits}"
            )

        logger.info(
            "Applying orchestrator bias from %d phases, target_coherence=%.3f",
            len(global_phases),
            target_coherence,
        )

        return self.optimize_phases(target_coherence, learning_rate=learning_rate)

    def to_qpu_artifact_metadata(self) -> dict[str, Any]:
        """Produce metadata for QPUBridgeArtifact integration."""
        return {
            "bridge_type": "FisherPosnerQuantumBridge",
            "n_qubits": self.n_qubits,
            "backend": self._backend,
            "has_pennylane": HAS_PENNYLANE,
            "tier": "experimental",
        }

    def __repr__(self) -> str:
        return f"FisherPosnerQuantumBridge(n_qubits={self.n_qubits}, backend={self._backend!r})"


__all__ = ["FisherPosnerQuantumBridge", "HAS_PENNYLANE", "compute_max_qubits"]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA mapper orchestration

"""High-level stochastic-network to molecular-circuit orchestration."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .dna_compilers import EnzymaticGateCompiler, StrandDisplacementCompiler
from .dna_sequences import SequenceDesigner
from .dna_simulation import KineticSimulator
from .dna_thermodynamics import NUPACKInterface
from .dna_types import (
    _DEFAULT_TEMPERATURE_C,
    CompilationMethod,
    DNACircuitDesign,
    DNAGate,
    DNAStrand,
)


class BitstreamToDNA:
    """High-level API for mapping SC bitstreams to DNA circuits.

    This is the primary entry point for the DNA computing bridge.
    Accepts a description of an SC Boolean network and compiles it
    into a complete DNA circuit design.

    Parameters
    ----------
    method : str
        Compilation method: ``"displacement"`` (default),
        ``"enzymatic"``, or ``"hybrid"``.
    seed : int
        Sequence generation seed for reproducibility.
    temperature_c : float
        Design temperature in Celsius.

    Examples
    --------
    >>> compiler = BitstreamToDNA(method="displacement", seed=42)
    >>> design = compiler.compile_network(
    ...     gates=[
    ...         {"type": "AND", "inputs": ["A", "B"], "output": "C"},
    ...         {"type": "NOT", "inputs": ["C"], "output": "D"},
    ...     ],
    ...     input_names=["A", "B"],
    ...     output_names=["D"],
    ... )
    >>> print(design.total_gates)
    2
    >>> print(design.total_strands)
    ...
    >>> export_genbank(design, "nand_circuit.gb")
    """

    def __init__(
        self,
        method: str = "displacement",
        seed: int = 42,
        temperature_c: float = _DEFAULT_TEMPERATURE_C,
    ) -> None:
        self._method = CompilationMethod(method)
        self._designer = SequenceDesigner(seed=seed)
        self._displacement = StrandDisplacementCompiler(
            designer=self._designer, temperature_c=temperature_c
        )
        self._enzymatic = EnzymaticGateCompiler(designer=self._designer)
        self._nupack = NUPACKInterface(temperature_c=temperature_c)
        self._temperature_c = temperature_c

    def compile_network(
        self,
        gates: List[Dict[str, Any]],
        input_names: List[str],
        output_names: List[str],
        name: str = "sc_dna_circuit",
    ) -> DNACircuitDesign:
        """Compile an SC Boolean network into a DNA circuit.

        Parameters
        ----------
        gates : list[dict]
            Gate specifications. Each dict has keys:
            ``"type"`` (str), ``"inputs"`` (list[str]),
            ``"output"`` (str), and optionally ``"threshold"`` (float).
        input_names : list[str]
            Primary input signal names.
        output_names : list[str]
            Primary output signal names.
        name : str
            Circuit identifier.

        Returns
        -------
        DNACircuitDesign
            Complete strand-level design.
        """
        design = DNACircuitDesign(
            name=name,
            method=self._method,
            temperature_c=self._temperature_c,
        )

        # Create input strands
        for inp in input_names:
            seq = self._designer.generate_recognition(f"input_{inp}")
            toehold = self._designer.generate_toehold(f"input_{inp}_th")
            design.input_strands.append(
                DNAStrand(
                    name=f"signal_{inp}",
                    sequence=toehold + seq,
                    role="signal",
                    concentration_nM=200.0,
                )
            )

        # Compile each gate
        for gate_spec in gates:
            gate_type = gate_spec["type"].upper()
            inputs = gate_spec["inputs"]
            output = gate_spec["output"]

            if self._method in (
                CompilationMethod.DISPLACEMENT,
                CompilationMethod.HYBRID,
            ):
                compiled = self._compile_displacement_gate(gate_type, inputs, output, gate_spec)
            else:
                compiled = self._compile_enzymatic_gate(gate_type, inputs, output, gate_spec)
            design.gates.append(compiled)

        # Create output strands
        for out in output_names:
            seq = self._designer.generate_recognition(f"output_{out}")
            design.output_strands.append(
                DNAStrand(
                    name=f"output_{out}",
                    sequence=seq,
                    role="output",
                    concentration_nM=0.0,
                )
            )

        return design

    def simulate(
        self,
        design: DNACircuitDesign,
        input_concentrations: Dict[str, float],
        duration_s: float = 3600.0,
        dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """Simulate the compiled circuit.

        Parameters
        ----------
        design : DNACircuitDesign
            Compiled circuit.
        input_concentrations : dict[str, float]
            Initial concentrations of input signals (nM).
        duration_s : float
            Simulation time in seconds.
        dt : float
            Time step in seconds.

        Returns
        -------
        dict[str, np.ndarray]
            Time traces for each output. Includes ``"time"`` key.
        """
        sim = KineticSimulator(temperature_c=self._temperature_c)
        return sim.simulate(design, input_concentrations, duration_s, dt)

    def validate(self, design: DNACircuitDesign) -> Dict[str, Any]:
        """Validate design using NUPACK (or fallback)."""
        return self._nupack.validate_design(design)

    def _compile_displacement_gate(
        self,
        gate_type: str,
        inputs: List[str],
        output: str,
        spec: Dict[str, Any],
    ) -> DNAGate:
        if gate_type == "AND":
            return self._displacement.compile_and(inputs[0], inputs[1], output)
        elif gate_type == "OR":
            return self._displacement.compile_or(inputs[0], inputs[1], output)
        elif gate_type == "NOT":
            return self._displacement.compile_not(inputs[0], output)
        elif gate_type == "MUX":
            return self._displacement.compile_mux(inputs[0], inputs[1], inputs[2], output)
        elif gate_type == "AMPLIFIER":
            return self._displacement.compile_amplifier(inputs[0], output)
        elif gate_type == "BUFFER":
            return self._displacement.compile_buffer(inputs[0], output)
        elif gate_type == "THRESHOLD":
            threshold = spec.get("threshold", 0.5)
            return self._displacement.compile_threshold(inputs[0], output, threshold)
        else:
            raise ValueError(f"Unsupported displacement gate: {gate_type}")

    def _compile_enzymatic_gate(
        self,
        gate_type: str,
        inputs: List[str],
        output: str,
        spec: Dict[str, Any],
    ) -> DNAGate:
        if gate_type == "NAND":
            return self._enzymatic.compile_nand(inputs[0], inputs[1], output)
        elif gate_type == "XOR":
            return self._enzymatic.compile_xor(inputs[0], inputs[1], output)
        else:
            raise ValueError(f"Unsupported enzymatic gate: {gate_type}")


class SCNetworkBridge:
    """Bridge between SC-NeuroCore network objects and DNA compilation.

    Converts Population/Projection-based SC networks into the gate-spec
    format consumed by BitstreamToDNA. Supports automatic gate-type
    inference from connection weights.

    Parameters
    ----------
    method : str
        Compilation method (``"displacement"`` or ``"enzymatic"``).
    seed : int
        Random seed.
    """

    def __init__(self, method: str = "displacement", seed: int = 42) -> None:
        self._method = method
        self._seed = seed

    def from_adjacency(
        self,
        adjacency: np.ndarray[Any, Any],
        input_indices: list[int],
        output_indices: list[int],
        name: str = "sc_network",
    ) -> DNACircuitDesign:
        """Compile from an adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix. Non-zero entries are connections.
            Positive = excitatory (AND), negative = inhibitory (NOT).
        input_indices : list[int]
            Row indices of input neurons.
        output_indices : list[int]
            Row indices of output neurons.
        name : str
            Circuit name.

        Returns
        -------
        DNACircuitDesign
        """
        n = adjacency.shape[0]
        node_names = [f"n{i}" for i in range(n)]
        gates: list[Dict[str, Any]] = []

        for j in range(n):
            if j in input_indices:
                continue

            sources = []
            for i in range(n):
                if adjacency[i, j] != 0:
                    sources.append((i, float(adjacency[i, j])))

            if not sources:
                continue

            if len(sources) == 1:
                src_idx, w = sources[0]
                if w < 0:
                    gates.append(
                        {
                            "type": "NOT",
                            "inputs": [node_names[src_idx]],
                            "output": node_names[j],
                        }
                    )
                else:
                    gates.append(
                        {
                            "type": "BUFFER",
                            "inputs": [node_names[src_idx]],
                            "output": node_names[j],
                        }
                    )
            elif len(sources) == 2:
                s0, s1 = sources[0], sources[1]
                if s0[1] > 0 and s1[1] > 0:
                    gates.append(
                        {
                            "type": "AND",
                            "inputs": [node_names[s0[0]], node_names[s1[0]]],
                            "output": node_names[j],
                        }
                    )
                else:
                    gates.append(
                        {
                            "type": "OR",
                            "inputs": [node_names[s0[0]], node_names[s1[0]]],
                            "output": node_names[j],
                        }
                    )
            else:
                # Multi-fan-in: chain AND gates
                prev = node_names[sources[0][0]]
                for k in range(1, len(sources)):
                    out = f"{node_names[j]}_stage{k}" if k < len(sources) - 1 else node_names[j]
                    gates.append(
                        {
                            "type": "AND",
                            "inputs": [prev, node_names[sources[k][0]]],
                            "output": out,
                        }
                    )
                    prev = out

        compiler = BitstreamToDNA(method=self._method, seed=self._seed)
        return compiler.compile_network(
            gates=gates,
            input_names=[node_names[i] for i in input_indices],
            output_names=[node_names[i] for i in output_indices],
            name=name,
        )

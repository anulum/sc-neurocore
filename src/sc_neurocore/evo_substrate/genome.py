# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary genome contracts and serialisation

"""Encode, fingerprint, and serialise evolutionary genomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class TopologyGene:
    """Encodes the network topology."""

    num_neurons: int = 16
    num_layers: int = 2
    connectivity: float = 0.3  # connection probability
    recurrent_fraction: float = 0.1
    bitstream_length: int = 256

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Serialise topology fields in their canonical five-value order."""
        return np.array(
            [
                self.num_neurons,
                self.num_layers,
                self.connectivity,
                self.recurrent_fraction,
                self.bitstream_length,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> TopologyGene:
        """Construct a topology gene while enforcing physical parameter bounds.

        Parameters
        ----------
        v
            Five values ordered as neuron count, layer count, connectivity,
            recurrent fraction, and bitstream length.
        """
        return cls(
            num_neurons=max(2, int(v[0])),
            num_layers=max(1, int(v[1])),
            connectivity=float(np.clip(v[2], 0.01, 1.0)),
            recurrent_fraction=float(np.clip(v[3], 0.0, 0.5)),
            bitstream_length=max(32, int(v[4])),
        )


@dataclass
class NeuronGene:
    """Encodes neuron-level parameters (ArcaneNeuron-compatible)."""

    tau_fast: float = 5.0
    tau_work: float = 200.0
    tau_deep: float = 10000.0
    theta: float = 1.0
    gamma: float = 0.2
    delta_conf: float = 0.3
    kappa: float = 5.0
    w_inh: float = 0.3

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Serialise neuron kinetics in their canonical eight-value order."""
        return np.array(
            [
                self.tau_fast,
                self.tau_work,
                self.tau_deep,
                self.theta,
                self.gamma,
                self.delta_conf,
                self.kappa,
                self.w_inh,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> NeuronGene:
        """Construct neuron kinetics while enforcing finite lower bounds.

        Parameters
        ----------
        v
            Eight values ordered as the documented neuron parameter block.
        """
        return cls(
            tau_fast=max(0.5, float(v[0])),
            tau_work=max(1.0, float(v[1])),
            tau_deep=max(10.0, float(v[2])),
            theta=max(0.1, float(v[3])),
            gamma=float(np.clip(v[4], 0.0, 1.0)),
            delta_conf=float(np.clip(v[5], 0.0, 1.0)),
            kappa=max(0.1, float(v[6])),
            w_inh=float(np.clip(v[7], 0.0, 1.0)),
        )


@dataclass
class PlasticityGene:
    """Encodes plasticity rule parameters."""

    stdp_lr: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stp_u_base: float = 0.5
    homeostatic_rate: float = 0.001
    meta_sensitivity: float = 1.0

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Serialise plasticity fields in their canonical six-value order."""
        return np.array(
            [
                self.stdp_lr,
                self.stdp_tau_plus,
                self.stdp_tau_minus,
                self.stp_u_base,
                self.homeostatic_rate,
                self.meta_sensitivity,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> PlasticityGene:
        """Construct plasticity parameters while enforcing valid rates.

        Parameters
        ----------
        v
            Six values ordered as the documented plasticity parameter block.
        """
        return cls(
            stdp_lr=max(1e-6, float(v[0])),
            stdp_tau_plus=max(1.0, float(v[1])),
            stdp_tau_minus=max(1.0, float(v[2])),
            stp_u_base=float(np.clip(v[3], 0.01, 0.99)),
            homeostatic_rate=max(1e-6, float(v[4])),
            meta_sensitivity=max(0.1, float(v[5])),
        )


@dataclass
class Genome:
    """Complete genome for an evolving SC organism."""

    genome_id: str = ""
    parent_id: str = ""
    generation: int = 0
    topology: TopologyGene = field(default_factory=TopologyGene)
    neuron: NeuronGene = field(default_factory=NeuronGene)
    plasticity: PlasticityGene = field(default_factory=PlasticityGene)
    weight_seed: int = 42
    identity_deep: float = 0.0

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Return the canonical 19-value topology-neuron-plasticity vector."""
        return np.concatenate(
            [
                self.topology.to_vector(),
                self.neuron.to_vector(),
                self.plasticity.to_vector(),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any], gen: int = 0) -> Genome:
        """Construct a genome from its canonical parameter vector.

        Parameters
        ----------
        v
            Nineteen-value canonical genome vector.
        gen
            Generation assigned to the reconstructed genome.
        """
        return cls(
            generation=gen,
            topology=TopologyGene.from_vector(v[0:5]),
            neuron=NeuronGene.from_vector(v[5:13]),
            plasticity=PlasticityGene.from_vector(v[13:19]),
        )

    @property
    def vector_dim(self) -> int:
        """Return the number of values in the canonical genome vector."""
        return len(self.to_vector())

    def compute_id(self) -> str:
        """Set and return the 12-hex SHA-256 prefix of the genome vector."""
        h = hashlib.sha256(self.to_vector().tobytes())
        self.genome_id = h.hexdigest()[:12]
        return self.genome_id


class GenomeSerializer:
    """Serializes/deserializes genomes for persistence."""

    @staticmethod
    def to_dict(genome: Genome) -> Dict[str, Any]:
        """Return a JSON-ready mapping that preserves genome identity fields."""
        return {
            "genome_id": genome.genome_id,
            "parent_id": genome.parent_id,
            "generation": genome.generation,
            "weight_seed": genome.weight_seed,
            "identity_deep": genome.identity_deep,
            "vector": genome.to_vector().tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Genome:
        """Reconstruct a genome from :meth:`to_dict` output."""
        v = np.array(d["vector"])
        g = Genome.from_vector(v, d.get("generation", 0))
        g.genome_id = d.get("genome_id", "")
        g.parent_id = d.get("parent_id", "")
        g.weight_seed = d.get("weight_seed", 42)
        g.identity_deep = d.get("identity_deep", 0.0)
        return g


__all__ = ["Genome", "GenomeSerializer", "NeuronGene", "PlasticityGene", "TopologyGene"]

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fluent builder for `ScGraph`

//! Fluent builder for `ScGraph`.

use crate::ir::graph::*;

/// Builder for constructing `ScGraph` instances.
pub struct ScGraphBuilder {
    graph: ScGraph,
}

impl ScGraphBuilder {
    /// Start building a new graph with the given module name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: ScGraph::new(name),
        }
    }

    /// Add a module input port.
    pub fn input(&mut self, name: impl Into<String>, ty: ScType) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Input {
            id,
            name: name.into(),
            ty,
        })
    }

    /// Add a module output port.
    pub fn output(&mut self, name: impl Into<String>, source: ValueId) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Output {
            id,
            name: name.into(),
            source,
        })
    }

    /// Add a compile-time constant.
    pub fn constant(&mut self, value: ScConst, ty: ScType) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Constant { id, value, ty })
    }

    /// Add a bitstream encode operation.
    pub fn encode(&mut self, prob: ValueId, length: usize, seed: u16) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Encode {
            id,
            prob,
            length,
            seed,
        })
    }

    /// Add a bitwise AND (stochastic multiply).
    pub fn bitwise_and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::BitwiseAnd { id, lhs, rhs })
    }

    /// Add a popcount operation.
    pub fn popcount(&mut self, input: ValueId) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Popcount { id, input })
    }

    /// Add a LIF neuron step.
    pub fn lif_step(
        &mut self,
        current: ValueId,
        leak: ValueId,
        gain: ValueId,
        noise: ValueId,
        params: LifParams,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::LifStep {
            id,
            current,
            leak,
            gain,
            noise,
            params,
        })
    }

    /// Add a dense SC layer forward pass.
    pub fn dense_forward(
        &mut self,
        inputs: ValueId,
        weights: ValueId,
        leak: ValueId,
        gain: ValueId,
        params: DenseParams,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::DenseForward {
            id,
            inputs,
            weights,
            leak,
            gain,
            params,
        })
    }

    /// Add a delay-coded learnable-spike layer forward pass.
    pub fn dcls_layer(
        &mut self,
        spike: ValueId,
        weights: ValueId,
        centre: ValueId,
        sigma: ValueId,
        params: DclsParams,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::DclsLayer {
            id,
            spike,
            weights,
            centre,
            sigma,
            params,
        })
    }

    /// Add a bitwise XOR (HDC binding).
    pub fn bitwise_xor(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::BitwiseXor { id, lhs, rhs })
    }

    /// Add a reduce operation (Sum or Max).
    pub fn reduce(&mut self, input: ValueId, mode: ReduceMode) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Reduce { id, input, mode })
    }

    /// Add a graph forward pass.
    pub fn graph_forward(
        &mut self,
        features: ValueId,
        adjacency: ValueId,
        n_nodes: usize,
        n_features: usize,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::GraphForward {
            id,
            features,
            adjacency,
            n_nodes,
            n_features,
        })
    }

    /// Add a softmax attention operation.
    pub fn softmax_attention(
        &mut self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        dim_k: usize,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph
            .push(ScOp::SoftmaxAttention { id, q, k, v, dim_k })
    }

    /// Add a Kuramoto integration step.
    pub fn kuramoto_step(
        &mut self,
        phases: ValueId,
        omega: ValueId,
        coupling: ValueId,
        dt: f64,
    ) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::KuramotoStep {
            id,
            phases,
            omega,
            coupling,
            dt,
        })
    }

    /// Add a scale (multiply by constant) operation.
    pub fn scale(&mut self, input: ValueId, factor: f64) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Scale { id, input, factor })
    }

    /// Add an offset (add constant) operation.
    pub fn offset(&mut self, input: ValueId, offset_val: f64) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::Offset {
            id,
            input,
            offset: offset_val,
        })
    }

    /// Add a constant integer division.
    pub fn div_const(&mut self, input: ValueId, divisor: u64) -> ValueId {
        let id = self.graph.fresh_id();
        self.graph.push(ScOp::DivConst { id, input, divisor })
    }

    /// Consume the builder and return the completed graph.
    pub fn build(self) -> ScGraph {
        self.graph
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - SystemVerilog target metadata

//! Target metadata and resource estimates for SystemVerilog emission.

use crate::ir::graph::{ScConst, ScGraph, ScOp};

/// Supported Zynq UltraScale+ MPSoC SKUs for the first target lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkuKind {
    Zu3eg,
    Zu9eg,
}

/// SystemVerilog emission target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvTarget {
    Generic,
    Zynq7 {
        device: String,
        clock_mhz: u32,
    },
    ZynqUltraScalePlus {
        sku: SkuKind,
        clock_mhz: u32,
        dsp_budget: u32,
        bram_36k_budget: u32,
        uram_budget: u32,
        prefer_uram_over_bram_above_bits: u64,
    },
}

/// Conservative compiler-side resource estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceReport {
    pub target_name: String,
    pub device_part: String,
    pub clock_mhz: u32,
    pub lut_estimated: u32,
    pub ff_estimated: u32,
    pub dsp_estimated: u32,
    pub bram_36k_estimated: u32,
    pub uram_estimated: u32,
    pub critical_path_estimate_ns: f64,
    pub dsp_budget: u32,
    pub bram_36k_budget: u32,
    pub uram_budget: u32,
    pub fits_dsp_budget: bool,
    pub fits_bram_budget: bool,
    pub fits_uram_budget: bool,
    pub dense_fold_plan: Option<DenseFoldPlan>,
}

/// Deterministic time-multiplexing plan for dense layers that exceed a target
/// one-DSP-per-MAC budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseFoldPlan {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub mac_count: u32,
    pub dsp_budget: u32,
    pub output_parallelism: u32,
    pub input_parallelism: u32,
    pub dsp_per_cycle: u32,
    pub input_fold_factor: u32,
    pub output_fold_factor: u32,
    pub compute_cycles: u32,
    pub fold_required: bool,
    pub fits_dsp_budget: bool,
}

impl SkuKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zu3eg => "ZU3EG",
            Self::Zu9eg => "ZU9EG",
        }
    }

    pub fn device_part(self) -> &'static str {
        match self {
            Self::Zu3eg => "xczu3eg-sbva484-1-e",
            Self::Zu9eg => "xczu9eg-ffvb1156-2-e",
        }
    }

    pub fn lut_budget(self) -> u32 {
        match self {
            Self::Zu3eg => 70_560,
            Self::Zu9eg => 274_080,
        }
    }

    pub fn ff_budget(self) -> u32 {
        match self {
            Self::Zu3eg => 141_120,
            Self::Zu9eg => 548_160,
        }
    }

    pub fn dsp_budget(self) -> u32 {
        match self {
            Self::Zu3eg => 360,
            Self::Zu9eg => 2_520,
        }
    }

    pub fn bram_36k_budget(self) -> u32 {
        match self {
            Self::Zu3eg => 216,
            Self::Zu9eg => 912,
        }
    }

    pub fn uram_budget(self) -> u32 {
        0
    }
}

impl SvTarget {
    pub fn zynq_ultrascale_plus(sku: SkuKind, clock_mhz: u32) -> Self {
        Self::ZynqUltraScalePlus {
            sku,
            clock_mhz,
            dsp_budget: sku.dsp_budget(),
            bram_36k_budget: sku.bram_36k_budget(),
            uram_budget: sku.uram_budget(),
            prefer_uram_over_bram_above_bits: 1_u64 << 20,
        }
    }

    pub fn target_name(&self) -> String {
        match self {
            Self::Generic => "generic".to_string(),
            Self::Zynq7 { device, .. } => format!("zynq7:{device}"),
            Self::ZynqUltraScalePlus { sku, .. } => {
                format!("zynq-ultrascale-plus:{}", sku.as_str())
            }
        }
    }

    pub fn clock_mhz(&self) -> u32 {
        match self {
            Self::Generic => 100,
            Self::Zynq7 { clock_mhz, .. } | Self::ZynqUltraScalePlus { clock_mhz, .. } => {
                *clock_mhz
            }
        }
    }

    pub fn device_part(&self) -> String {
        match self {
            Self::Generic => "generic".to_string(),
            Self::Zynq7 { device, .. } => device.clone(),
            Self::ZynqUltraScalePlus { sku, .. } => sku.device_part().to_string(),
        }
    }

    pub fn dsp_primitive(&self) -> &'static str {
        match self {
            Self::ZynqUltraScalePlus { .. } => "DSP48E2",
            Self::Generic | Self::Zynq7 { .. } => "generic",
        }
    }

    pub fn dsp_attribute(&self) -> Option<&'static str> {
        match self {
            Self::ZynqUltraScalePlus { .. } => Some(
                "(* use_dsp = \"yes\", sc_target_dsp = \"DSP48E2\", sc_target_family = \"zynq_ultrascale_plus\" *)",
            ),
            Self::Generic | Self::Zynq7 { .. } => None,
        }
    }

    pub fn ram_style_for_bits(&self, bits: u64) -> Option<&'static str> {
        match self {
            Self::ZynqUltraScalePlus {
                uram_budget,
                prefer_uram_over_bram_above_bits,
                ..
            } if *uram_budget > 0 && bits >= *prefer_uram_over_bram_above_bits => Some("ultra"),
            Self::ZynqUltraScalePlus { .. } if bits >= 1_024 => Some("block"),
            Self::ZynqUltraScalePlus { .. } => Some("distributed"),
            Self::Generic | Self::Zynq7 { .. } => None,
        }
    }

    pub fn dense_fold_plan(&self, n_inputs: usize, n_outputs: usize) -> Option<DenseFoldPlan> {
        let dsp_budget = match self {
            Self::ZynqUltraScalePlus { dsp_budget, .. } => *dsp_budget,
            Self::Generic | Self::Zynq7 { .. } => return None,
        };
        Some(plan_dense_fold(n_inputs, n_outputs, dsp_budget))
    }

    pub fn header_comment(&self) -> String {
        match self {
            Self::Generic => String::new(),
            Self::Zynq7 { device, clock_mhz } => format!(
                "// Target: Zynq-7 device={device}, clock={clock_mhz} MHz\n\n"
            ),
            Self::ZynqUltraScalePlus { sku, clock_mhz, .. } => format!(
                "// Target: Zynq UltraScale+ MPSoC {}, part={}, clock={} MHz\n// DSP primitive: DSP48E2\n\n",
                sku.as_str(),
                sku.device_part(),
                clock_mhz
            ),
        }
    }

    pub fn estimate_graph(&self, graph: &ScGraph) -> ResourceReport {
        let mut lut_estimated = 128_u32;
        let mut ff_estimated = 128_u32;
        let mut dsp_estimated = 0_u32;
        let mut bram_bits = 0_u64;
        let uram_bits = 0_u64;
        let mut critical_path_estimate_ns = 2.5_f64;
        let mut dense_fold_plan: Option<DenseFoldPlan> = None;

        for op in &graph.ops {
            match op {
                ScOp::DenseForward { params, .. } => {
                    let macs = saturating_u32(params.n_inputs.saturating_mul(params.n_neurons));
                    dsp_estimated = dsp_estimated.saturating_add(macs);
                    if let Some(plan) = self.dense_fold_plan(params.n_inputs, params.n_neurons) {
                        if plan.fold_required {
                            dense_fold_plan = Some(plan);
                        }
                    }
                    lut_estimated = lut_estimated.saturating_add(220).saturating_add(macs * 6);
                    ff_estimated = ff_estimated.saturating_add(180).saturating_add(macs * 4);
                    bram_bits = bram_bits.saturating_add(
                        (params.n_inputs as u64)
                            .saturating_mul(params.n_neurons as u64)
                            .saturating_mul(params.data_width as u64),
                    );
                    critical_path_estimate_ns = critical_path_estimate_ns.max(4.0);
                }
                ScOp::DclsLayer { params, .. } => {
                    let taps = saturating_u32(params.n_taps);
                    dsp_estimated = dsp_estimated.saturating_add(taps);
                    lut_estimated = lut_estimated.saturating_add(320).saturating_add(taps * 48);
                    ff_estimated = ff_estimated.saturating_add(220).saturating_add(taps * 32);
                    bram_bits = bram_bits.saturating_add(
                        (params.delay_depth as u64).saturating_mul(params.n_taps as u64),
                    );
                    critical_path_estimate_ns = critical_path_estimate_ns.max(4.5);
                }
                ScOp::LifStep { .. } => {
                    dsp_estimated = dsp_estimated.saturating_add(2);
                    lut_estimated = lut_estimated.saturating_add(180);
                    ff_estimated = ff_estimated.saturating_add(96);
                    critical_path_estimate_ns = critical_path_estimate_ns.max(3.2);
                }
                ScOp::KuramotoStep { .. } => {
                    dsp_estimated = dsp_estimated.saturating_add(4);
                    lut_estimated = lut_estimated.saturating_add(512);
                    ff_estimated = ff_estimated.saturating_add(256);
                    critical_path_estimate_ns = critical_path_estimate_ns.max(5.0);
                }
                ScOp::Constant { value, .. } => {
                    bram_bits = bram_bits.saturating_add(constant_bits(value));
                }
                ScOp::Input { .. }
                | ScOp::Output { .. }
                | ScOp::Encode { .. }
                | ScOp::BitwiseAnd { .. }
                | ScOp::Popcount { .. }
                | ScOp::BitwiseXor { .. }
                | ScOp::Reduce { .. }
                | ScOp::GraphForward { .. }
                | ScOp::SoftmaxAttention { .. }
                | ScOp::Scale { .. }
                | ScOp::Offset { .. }
                | ScOp::DivConst { .. } => {
                    lut_estimated = lut_estimated.saturating_add(16);
                    ff_estimated = ff_estimated.saturating_add(8);
                }
            }
        }

        let bram_36k_estimated = ceil_div_u64(bram_bits, 36_864).min(u64::from(u32::MAX)) as u32;
        let uram_estimated = ceil_div_u64(uram_bits, 294_912).min(u64::from(u32::MAX)) as u32;
        let (dsp_budget, bram_budget, uram_budget) = match self {
            Self::ZynqUltraScalePlus {
                dsp_budget,
                bram_36k_budget,
                uram_budget,
                ..
            } => (*dsp_budget, *bram_36k_budget, *uram_budget),
            Self::Zynq7 { .. } | Self::Generic => (u32::MAX, u32::MAX, u32::MAX),
        };

        ResourceReport {
            target_name: self.target_name(),
            device_part: self.device_part(),
            clock_mhz: self.clock_mhz(),
            lut_estimated,
            ff_estimated,
            dsp_estimated,
            bram_36k_estimated,
            uram_estimated,
            critical_path_estimate_ns,
            dsp_budget,
            bram_36k_budget: bram_budget,
            uram_budget,
            fits_dsp_budget: dsp_estimated <= dsp_budget,
            fits_bram_budget: bram_36k_estimated <= bram_budget,
            fits_uram_budget: uram_estimated <= uram_budget,
            dense_fold_plan,
        }
    }
}

fn plan_dense_fold(n_inputs: usize, n_outputs: usize, dsp_budget: u32) -> DenseFoldPlan {
    let mac_count = saturating_u32(n_inputs.saturating_mul(n_outputs));
    if n_inputs == 0 || n_outputs == 0 || dsp_budget == 0 {
        return DenseFoldPlan {
            n_inputs,
            n_outputs,
            mac_count,
            dsp_budget,
            output_parallelism: 0,
            input_parallelism: 0,
            dsp_per_cycle: 0,
            input_fold_factor: 0,
            output_fold_factor: 0,
            compute_cycles: 0,
            fold_required: mac_count > dsp_budget,
            fits_dsp_budget: dsp_budget >= 1,
        };
    }

    let n_inputs_u32 = saturating_u32(n_inputs).max(1);
    let n_outputs_u32 = saturating_u32(n_outputs).max(1);
    let output_parallelism = if dsp_budget >= n_inputs_u32 {
        (dsp_budget / n_inputs_u32).clamp(1, n_outputs_u32)
    } else {
        1
    };
    let input_parallelism = (dsp_budget / output_parallelism).clamp(1, n_inputs_u32);
    let dsp_per_cycle = output_parallelism.saturating_mul(input_parallelism);
    let input_fold_factor = ceil_div_u64(n_inputs_u32 as u64, input_parallelism as u64) as u32;
    let output_fold_factor = ceil_div_u64(n_outputs_u32 as u64, output_parallelism as u64) as u32;
    let compute_cycles = input_fold_factor.saturating_mul(output_fold_factor);
    DenseFoldPlan {
        n_inputs,
        n_outputs,
        mac_count,
        dsp_budget,
        output_parallelism,
        input_parallelism,
        dsp_per_cycle,
        input_fold_factor,
        output_fold_factor,
        compute_cycles,
        fold_required: mac_count > dsp_budget,
        fits_dsp_budget: dsp_per_cycle <= dsp_budget,
    }
}

fn constant_bits(value: &ScConst) -> u64 {
    match value {
        ScConst::F64(_) | ScConst::I64(_) => 16,
        ScConst::U64(_) => 32,
        ScConst::F64Vec(values) => values.len() as u64 * 16,
        ScConst::I64Vec(values) => values.len() as u64 * 16,
    }
}

fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    if value == 0 {
        0
    } else {
        1 + (value - 1) / divisor
    }
}

fn saturating_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::ScGraphBuilder;
    use crate::ir::graph::{DenseParams, ScConst, ScType};

    #[test]
    fn ultrascale_plus_sku_budgets_match_target_baseline_table() {
        assert_eq!(SkuKind::Zu3eg.dsp_budget(), 360);
        assert_eq!(SkuKind::Zu3eg.bram_36k_budget(), 216);
        assert_eq!(SkuKind::Zu3eg.uram_budget(), 0);
        assert_eq!(SkuKind::Zu9eg.dsp_budget(), 2_520);
        assert_eq!(SkuKind::Zu9eg.bram_36k_budget(), 912);
        assert_eq!(SkuKind::Zu9eg.uram_budget(), 0);
    }

    #[test]
    fn ultrascale_plus_uses_device_family_dsp_primitive() {
        let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
        assert_eq!(target.dsp_primitive(), "DSP48E2");
        assert!(target.dsp_attribute().unwrap().contains("DSP48E2"));
        assert!(!target.dsp_attribute().unwrap().contains(&format!("DSP{}", 58)));
    }

    #[test]
    fn ultrascale_plus_resource_report_tracks_dense_mac_budget() {
        let mut builder = ScGraphBuilder::new("resource_dense");
        let inputs = builder.input(
            "inputs",
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
                count: 4,
            },
        );
        let weights = builder.constant(
            ScConst::I64Vec(vec![128; 12]),
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
                count: 12,
            },
        );
        let leak = builder.constant(ScConst::I64(16), ScType::FixedPoint { width: 16, frac: 8 });
        let gain = builder.constant(ScConst::I64(1), ScType::FixedPoint { width: 16, frac: 8 });
        let dense = builder.dense_forward(
            inputs,
            weights,
            leak,
            gain,
            DenseParams {
                n_inputs: 4,
                n_neurons: 3,
                ..DenseParams::default()
            },
        );
        builder.output("spikes", dense);
        let graph = builder.build();
        let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
        let report = target.estimate_graph(&graph);
        assert!(report.dsp_estimated >= 12);
        assert!(report.bram_36k_estimated <= report.bram_36k_budget);
        assert!(report.fits_dsp_budget);
        assert_eq!(report.device_part, "xczu3eg-sbva484-1-e");
    }

    #[test]
    fn dense_fold_plan_maps_shd_scale_dense_into_zu3eg_budget() {
        let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
        let plan = target
            .dense_fold_plan(64, 32)
            .expect("UltraScale+ target must expose dense folding");
        assert_eq!(plan.mac_count, 2_048);
        assert_eq!(plan.dsp_budget, 360);
        assert_eq!(plan.output_parallelism, 5);
        assert_eq!(plan.input_parallelism, 64);
        assert_eq!(plan.dsp_per_cycle, 320);
        assert_eq!(plan.output_fold_factor, 7);
        assert_eq!(plan.input_fold_factor, 1);
        assert_eq!(plan.compute_cycles, 7);
        assert!(plan.fold_required);
        assert!(plan.fits_dsp_budget);
    }

    #[test]
    fn resource_report_carries_dense_fold_plan_when_unfurled_budget_fails() {
        let mut builder = ScGraphBuilder::new("folded_resource_dense");
        let inputs = builder.input(
            "inputs",
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
                count: 64,
            },
        );
        let weights = builder.constant(
            ScConst::I64Vec(vec![128; 64 * 32]),
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
                count: 64 * 32,
            },
        );
        let leak = builder.constant(ScConst::I64(16), ScType::FixedPoint { width: 16, frac: 8 });
        let gain = builder.constant(ScConst::I64(1), ScType::FixedPoint { width: 16, frac: 8 });
        let dense = builder.dense_forward(
            inputs,
            weights,
            leak,
            gain,
            DenseParams {
                n_inputs: 64,
                n_neurons: 32,
                ..DenseParams::default()
            },
        );
        builder.output("spikes", dense);
        let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
        let report = target.estimate_graph(&builder.build());
        let plan = report
            .dense_fold_plan
            .as_ref()
            .expect("over-budget dense graph must carry fold plan");
        assert!(!report.fits_dsp_budget);
        assert_eq!(plan.dsp_per_cycle, 320);
        assert!(plan.fits_dsp_budget);
    }
}

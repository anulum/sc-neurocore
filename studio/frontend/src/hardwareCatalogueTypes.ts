// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio hardware registration wire contract

/** Declared NIR planning constraints, without a vendor execution claim. */
export interface NirHardwareProfile {
  target_id: string;
  display_name: string;
  backend_status: string;
  supported_nir_nodes: string[];
  unsupported_nir_nodes: string[];
  sc_constraints: {
    bitstream_lengths: number[];
    stream_transport: string;
    precision_modes: string[];
    stochastic_sources: string[];
    back_annotation_channels: string[];
  };
  notes: string[];
}

/** The configured target and its capacity reference, without a run verdict. */
export interface HardwareSynthesisProfile {
  target: string;
  capacity_device: string;
  capacity: Record<string, number>;
  synthesis_command: string | null;
  routing_tool: string | null;
  device_selector: string | null;
  package_selector: string | null;
}

/** One registration; physical receipts are separate from declared interfaces. */
export interface HardwareCatalogueEntry {
  target_id: string;
  name: string;
  kind: "neuromorphic" | "fpga" | "board";
  registration_sources: string[];
  planning_profile: boolean;
  nir_profile: NirHardwareProfile | null;
  adapter_package: boolean;
  synthesis_profile: HardwareSynthesisProfile | null;
  driver: string | null;
  requirements: string[];
  limitations: string[];
  evidence_level: "registration";
  physical_evidence: string[];
}

/** Public discovery response. It neither probes nor programs a device. */
export interface HardwareCatalogueResponse {
  schema_version: "studio.hardware-catalogue.v1";
  entries: HardwareCatalogueEntry[];
  evidence_scope: string;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { BenchmarkSubmission } from "../api/client";
import { contributionPreview } from "./BenchmarkContribution";

const submission: BenchmarkSubmission = {
  schema_version: "scpn.benchmark.submission.v1",
  kernel: "dcls_max_forward_batch_q88",
  workload: { n_channels: 512, n_taps: 32, elements: 16384, spike_density: 0.5 },
  backends: [
    { backend: "rust", median_call_ms: 1.3, channels_per_s: 1, speedup_over_python: 85, repeats: 12, bit_exact: true },
    { backend: "python", median_call_ms: 112, channels_per_s: 1, speedup_over_python: 1, repeats: 12, bit_exact: true },
  ],
  parity: { reference: "python", tolerance: 0, bit_exact_all: true },
  environment: { cpu: "Test CPU", os: "Linux 6", python: "3.12.3", numpy: "2.2", toolchains: {} },
  hardware_measurement_claimed: false,
  contributor: { handle: "" },
};

describe("contributionPreview", () => {
  it("shows the aggregatable facts the user is about to send", () => {
    const preview = contributionPreview(submission);
    expect(preview).toContain("Test CPU");
    expect(preview).toContain("rust");
    expect(preview).toContain("speedup_over_python");
    expect(preview).toContain("bit_exact_all");
  });

  it("never includes machine-identifying fields", () => {
    const preview = contributionPreview(submission).toLowerCase();
    for (const forbidden of ["hostname", "username", "ip", "mac", "machine_id"]) {
      expect(preview).not.toContain(forbidden);
    }
  });
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  capabilityRegistryContract,
  compilerCapability,
  defaultApiMocks,
  installApiDispatcher,
  operatorStatus,
  registry,
  synthesisCapability,
  type ApiMockPayload,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
});

test("synthesis dashboard renders target provenance matrix from all-target run", async ({ page }) => {
  const verilog = "module test(input clk, output y); assign y = clk; endmodule";
  const synthesisJobList = {
    jobs: [
      {
        artifacts: [
          {
            relative_path: "synthesis/multi-target-result.json",
            sha256: "d".repeat(64),
            size_bytes: 512,
          },
          {
            relative_path: "synthesis/multi-target-evidence.json",
            sha256: "e".repeat(64),
            size_bytes: 384,
          },
        ],
        created_at_utc: "2026-06-21T12:00:00Z",
        error: null,
        execution_model: "process",
        finished_at_utc: "2026-06-21T12:00:01Z",
        job_id: "sj_synthesis",
        kind: "synthesis",
        owner: "studio-synthesis",
        request_id: "req-synth",
        result: null,
        started_at_utc: "2026-06-21T12:00:00Z",
        status: "completed",
      },
    ],
    schema_version: "studio.jobs.list.v1",
  };
  const api = await installApiDispatcher(
    page,
    new Map<string, ApiMockPayload>([
      [
        "/api/studio/capabilities",
        registry([capabilityRegistryContract, compilerCapability, synthesisCapability]),
      ],
      ["/api/models", []],
      ["/api/templates", []],
      ["/api/presets", []],
      ["/api/synth/tools-status", {
        nextpnr_ice40: { available: false, version: null },
        yosys: { available: true, version: "Yosys 0.test" },
      }],
      ["/api/ir/emit-sv-direct", {
        chars: verilog.length,
        compile_traceability: {
          evidence_classification: "compile",
          input_sha256: "1".repeat(64),
          output: {
            language: "systemverilog",
            module_name: "test",
            rtl_chars: verilog.length,
            rtl_sha256: "2".repeat(64),
          },
          schema_version: "studio.compile-traceability.v1",
          source: "ode",
          source_payload: {
            equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
            init: { v: -65 },
            params: { C: 1, E_L: -65, tau_m: 10 },
            reset: "v = -65",
            threshold: "v > -50",
          },
          traceability_sha256: "3".repeat(64),
        },
        ir_repr: "%0 = input clk",
        module_name: "test",
        verilog,
      }],
      ["/api/studio/evidence/bundle", {
        sequence: [
          {
            artifact_paths: [
              "evidence/replay.json",
              "evidence/manifest.json",
            ],
            artifacts: [
              {
                relative_path: "evidence/replay.json",
                sha256: "c".repeat(64),
                size_bytes: 128,
              },
            ],
            bundle_id: "seb_compile",
            job_id: "sj_compile",
            manifest: {
              entries: [
                { type: "command_replay" },
                { type: "manifest" },
              ],
            },
            schema_version: "studio.evidence-bundle.v1",
            summary: {
              artifact_path_count: 2,
              entry_count: 2,
              entry_type_counts: {
                command_replay: 1,
                manifest: 1,
              },
              evidence_classification_counts: {},
              source_job_count: 0,
              source_job_kind_counts: {},
              source_job_owner_counts: {},
            },
          },
          {
            artifact_paths: [
              "evidence/jobs/sj_synthesis/record.json",
              "evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-result.json",
              "evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json",
              "evidence/manifest.json",
            ],
            artifacts: [
              {
                relative_path: "evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json",
                sha256: "f".repeat(64),
                size_bytes: 384,
              },
            ],
            bundle_id: "seb_synthesis",
            job_id: "sj_synthesis_bundle",
            manifest: {
              entries: [
                { type: "source_job_record" },
                { type: "action_evidence" },
              ],
            },
            schema_version: "studio.evidence-bundle.v1",
            summary: {
              artifact_path_count: 4,
              entry_count: 4,
              entry_type_counts: {
                action_evidence: 1,
                manifest: 1,
                source_job_artifact: 2,
              },
              evidence_classification_counts: {
                synthesis: 1,
              },
              source_job_count: 1,
              source_job_kind_counts: {
                synthesis: 1,
              },
              source_job_owner_counts: {
                "studio-synthesis": 1,
              },
            },
          },
        ],
      }],
      ["/api/studio/jobs/sj_compile/artifacts/evidence/replay.json", {
        binaryBody: "{\"kind\":\"compile-replay\"}\n",
        contentType: "application/json",
      }],
      ["/api/studio/jobs/sj_synthesis_bundle/artifacts/evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json", {
        binaryBody: "{\"kind\":\"synthesis-evidence\"}\n",
        contentType: "application/json",
      }],
      ["/api/studio/operator/status", operatorStatus],
      ["/api/studio/auth/session", {
        authenticated: true,
        principal_id: "svc-admin",
        roles: ["studio.admin", "studio.viewer"],
      }],
      ["/api/studio/jobs", synthesisJobList],
      ["/api/synth/multi-target", {
        supported: ["ice40", "gowin"],
        target_provenance_matrix: {
          matrix_sha256: "a".repeat(64),
          schema_version: "studio.synthesis-target-provenance-matrix.v1",
          targets: {
            gowin: {
              capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
              device: null,
              evidence_classification: "synthesis",
              pnr_ready: true,
              pnr_tool: null,
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_gowin",
              synthesis_ready: true,
              target: "gowin",
              tools: [
                {
                  available: true,
                  executable: "yosys",
                  key: "yosys",
                  role: "synthesis",
                  version: "Yosys 0.test",
                },
              ],
            },
            ice40: {
              capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
              device: "up5k",
              evidence_classification: "synthesis",
              pnr_ready: false,
              pnr_tool: "nextpnr-ice40",
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_ice40",
              synthesis_ready: true,
              target: "ice40",
              tools: [
                {
                  available: true,
                  executable: "yosys",
                  key: "yosys",
                  role: "synthesis",
                  version: "Yosys 0.test",
                },
                {
                  available: false,
                  executable: "nextpnr-ice40",
                  key: "nextpnr_ice40",
                  role: "place_and_route",
                  version: null,
                },
              ],
            },
          },
        },
        targets: {
          gowin: {
            capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
            log_excerpt: "",
            resources: { brams: 0, cells: 1, dsps: 0, ffs: 1, luts: 2, wires: 1 },
            success: true,
            target: "gowin",
            target_provenance: {
              capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
              device: null,
              evidence_classification: "synthesis",
              pnr_ready: true,
              pnr_tool: null,
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_gowin",
              synthesis_ready: true,
              target: "gowin",
              tools: [],
            },
            utilisation: { brams: 0, dsps: 0, ffs: 0, luts: 0 },
          },
          ice40: {
            capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
            log_excerpt: "",
            resources: { brams: 0, cells: 1, dsps: 0, ffs: 1, luts: 2, wires: 1 },
            success: true,
            target: "ice40",
            target_provenance: {
              capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
              device: "up5k",
              evidence_classification: "synthesis",
              pnr_ready: false,
              pnr_tool: "nextpnr-ice40",
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_ice40",
              synthesis_ready: true,
              target: "ice40",
              tools: [],
            },
            utilisation: { brams: 0, dsps: 0, ffs: 0, luts: 0 },
          },
        },
      }],
    ]),
  );

  await page.goto("/");
  await page.getByRole("button", { name: "ODE", exact: true }).click();
  await page.getByRole("button", { name: "SV", exact: true }).click();
  await expect(page.getByText("SystemVerilog")).toBeVisible();
  await expect(page.getByText("trace 333333333333")).toBeVisible();
  await page.getByRole("button", { name: "Export compile evidence bundle" }).click();
  await expect(page.getByText("bundle seb_compile")).toBeVisible();
  await expect(page.getByText("evidence/replay.json", { exact: true })).toBeVisible();
  await expect(page.getByText("128 B - sha cccccccccccc")).toBeVisible();

  const evidenceBodies = api.bodies("/api/studio/evidence/bundle");
  expect(evidenceBodies).toHaveLength(1);
  expect(evidenceBodies[0]).toMatchObject({
    command_replay: {
      method: "POST",
      request_sha256: "1".repeat(64),
      route: "/api/ir/emit-sv-direct",
    },
    include_audit: true,
    project_name: null,
  });
  await page
    .getByRole("button", { name: "Download compile evidence artifact evidence/replay.json" })
    .click();
  const compileArtifactPath = "/api/studio/jobs/sj_compile/artifacts/evidence/replay.json";
  // The click starts the download and does not wait for it; polling asserts
  // the request was made rather than that it had already been made.
  await expect.poll(() => api.requests(compileArtifactPath)).toBe(1);
  expect(api.headers(compileArtifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });

  await page.getByRole("button", { name: "FPGA" }).first().click();
  await page.getByRole("button", { name: "All Targets" }).click();

  await expect(page.getByText("Target provenance matrix")).toBeVisible();
  await expect(page.getByText("aaaaaaaaaaaa")).toBeVisible();
  const matrixTable = page.getByRole("table").nth(1);
  await expect(matrixTable.getByRole("cell", { exact: true, name: "ICE40" })).toBeVisible();
  await expect(matrixTable.getByRole("cell", { exact: true, name: "up5k" })).toBeVisible();
  await expect(matrixTable.getByText("missing - nextpnr-ice40 missing")).toBeVisible();
  await expect(matrixTable.getByText("ready - not required")).toBeVisible();

  await page.getByRole("button", { name: "Export synthesis evidence bundle" }).click();
  await expect(page.getByText("bundle seb_synthesis")).toBeVisible();
  await expect(page.getByText(
    "evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json",
    { exact: true },
  )).toBeVisible();
  await expect(page.getByText("384 B - sha ffffffffffff")).toBeVisible();

  const finalEvidenceBodies = api.bodies("/api/studio/evidence/bundle");
  expect(finalEvidenceBodies).toHaveLength(2);
  expect(finalEvidenceBodies[1]).toMatchObject({
    command_replay: null,
    include_audit: true,
    job_ids: ["sj_synthesis"],
    project_name: null,
  });
  const synthesisArtifactPath = "/api/studio/jobs/sj_synthesis_bundle/artifacts/evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json";
  await page
    .getByRole("button", {
      name: "Download synthesis evidence artefact evidence/jobs/sj_synthesis/artifacts/synthesis/multi-target-evidence.json",
    })
    .click();
  // The click starts the download and does not wait for it; polling asserts
  // the request was made rather than that it had already been made.
  await expect.poll(() => api.requests(synthesisArtifactPath)).toBe(1);
  expect(api.headers(synthesisArtifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });
});

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  artifactJobList,
  defaultApiMocks,
  installApiDispatcher,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
});

test("admin evidence bundle form submits simulation and analysis result payloads", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json", {
    binaryBody: "{\"kind\":\"simulation\"}\n",
    contentType: "application/json",
  });
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  const simulationPayload = {
    dt: 0.1,
    n_steps: 2,
    run_metadata: {
      dt: 0.1,
      evidence_classification: "simulation",
      input_sha256: "1".repeat(64),
      n_steps: 2,
      result_sha256: "2".repeat(64),
      sample_count: 2,
      schema_version: "studio.simulation-run.v1",
      source: "ode",
      spike_count: 0,
      state_variables: ["v"],
    },
    spike_count: 0,
    states: { v: [0, 0.1] },
    time: [0, 0.1],
  };
  const analysisPayload = {
    analysis_metadata: {
      analysis_type: "fi_curve",
      evidence_classification: "analysis",
      input_sha256: "3".repeat(64),
      output_keys: ["currents", "rates"],
      result_sha256: "4".repeat(64),
      schema_version: "studio.analysis-result.v1",
      source: "ode",
    },
    currents: [0, 1],
    rates: [0, 10],
  };
  const defaultFlowRunPayload = {
    action_order: ["auto_tune_adaptive_precision"],
    executed_count: 1,
    execution_time_ms: 1,
    flow_id: "studio_default_adaptive_precision_v1",
    preset_id: "fpga_precision",
    reproducibility_manifest: {
      hash_algorithm: "sha256",
      inputs_fingerprint_sha256: "7".repeat(64),
      run_fingerprint_sha256: "8".repeat(64),
    },
    results: [],
    schema_version: "sc-neurocore.studio.default-flow-run.v1",
  };
  const defaultFlowAttestationPayload = {
    attestation_fingerprint_sha256: "9".repeat(64),
    flow_id: "studio_default_adaptive_precision_v1",
    inputs_fingerprint_sha256: "7".repeat(64),
    plan_fingerprint_sha256: "a".repeat(64),
    preset_id: "fpga_precision",
    run_fingerprint_sha256: "8".repeat(64),
    schema_version: "sc-neurocore.studio.default-flow-attestation.v1",
  };

  await page.getByRole("textbox", { name: "Evidence simulation JSON" }).fill(
    JSON.stringify(simulationPayload),
  );
  await page.getByRole("textbox", { name: "Evidence analysis JSON" }).fill(
    JSON.stringify(analysisPayload),
  );
  await page.getByRole("textbox", { name: "Evidence default-flow run JSON" }).fill(
    JSON.stringify(defaultFlowRunPayload),
  );
  await page.getByRole("textbox", { name: "Evidence default-flow attestation JSON" }).fill(
    JSON.stringify(defaultFlowAttestationPayload),
  );
  await page.getByRole("button", { name: "Create evidence bundle" }).click();

  await expect(page.getByText("seb_sj_browser")).toBeVisible();
  await expect(page.getByText("analysis_result:1")).toBeVisible();
  await expect(page.getByText("simulation:1")).toBeVisible();
  await expect(page.getByText("simulation - evidence/simulations/000.json")).toBeVisible();
  await expect(page.getByText("analysis - evidence/analyses/000.json")).toBeVisible();
  await expect(page.getByText("unclassified - sha ffffffffffff")).toBeVisible();
  await expect(page.getByText("evidence/simulations/000.json", { exact: true })).toBeVisible();
  await expect(page.getByText("256 B - sha cccccccccccc")).toBeVisible();

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    analysis_results: [analysisPayload],
    default_flow_attestations: [defaultFlowAttestationPayload],
    default_flow_runs: [defaultFlowRunPayload],
    include_audit: true,
    simulation_results: [simulationPayload],
  });

  await page
    .getByRole("button", { name: "Download evidence artifact evidence/simulations/000.json" })
    .click();
  const artifactPath = "/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json";
  // The click starts the download and does not wait for it; polling asserts
  // the request was made rather than that it had already been made.
  await expect.poll(() => api.requests(artifactPath)).toBe(1);
  expect(api.headers(artifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });
});

test("admin job rows can seed evidence bundle job IDs", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/jobs", artifactJobList);
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  await expect(page.getByText("compiler - sj_artifact")).toBeVisible();
  await expect(page.getByText("2 artifacts - 1 evidence")).toBeVisible();
  await expect(page.getByText("reports/result.txt, compiler/compile-evidence.json")).toBeVisible();

  await page.getByRole("button", { name: "Add sj_artifact to evidence bundle" }).click();
  await expect(page.getByRole("textbox", { name: "Evidence job IDs" })).toHaveValue("sj_artifact");

  await page.getByRole("button", { name: "Create evidence bundle" }).click();
  await expect(page.getByText("seb_sj_browser")).toBeVisible();

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    include_audit: true,
    job_ids: ["sj_artifact"],
  });
});

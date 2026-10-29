// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted scientific result presentation tests
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import HostedJobResult, { hostedCompilerView } from "./HostedJobResult";
import type { HostJson } from "./hostApiTypes";

let root: Root, host: HTMLDivElement;
const accepted = { accepted: true, result: { ir_text: "native graph <script>text</script>",
  systemverilog: "module actual_native; endmodule", valid: true, source_replay: "passed" }, refusal: null };

/** Bind a DOM subject to the already-verified result-document input contract. */
function document(value: unknown): HostJson {
  const json = JSON.stringify(value);
  return { json, digest: `sha256:${"1".repeat(64)}`, bytes: new TextEncoder().encode(json).length };
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  host = window.document.createElement("div"); window.document.body.append(host); root = createRoot(host);
});
afterEach(async () => {
  await act(async () => { root.unmount(); }); window.document.body.replaceChildren(); vi.unstubAllGlobals();
});

it("renders actual compiler verification/replay and requests exact result/graph/RTL downloads", async () => {
  const selected: ({ digest: string } | { text: string })[] = [], data = document(accepted);
  await act(async () => { root.render(<HostedJobResult document={data} jobId="sj_1234567890abcdef" busy={false}
    onDownload={(selection) => { selected.push(selection); }} />); });
  expect(host.textContent).toContain("Graph structure: verified. Source replay: passed.");
  expect(host.textContent).toContain(data.json); expect(host.querySelector("script")).toBeNull();
  for (const button of host.querySelectorAll("button")) await act(async () => { button.click(); });
  expect(selected).toEqual([{ digest: data.digest }, { text: accepted.result.ir_text }, { text: accepted.result.systemverilog }]);
});

it("keeps a real compiler refusal separate from job transport completion", async () => {
  await act(async () => { root.render(<HostedJobResult document={document({ accepted: false, result: null,
    refusal: { code: "invalid_input" } })} jobId="sj_1234567890abcdef" busy={false} onDownload={() => undefined} />); });
  expect(host.textContent).toContain("Compiler refused the operation");
  expect(host.querySelectorAll("button")).toHaveLength(1);
  expect(host.textContent).not.toContain("Graph structure: verified");
});

it("shows failed structural verification truthfully and disables downloads during another request", async () => {
  const input = document({ ...accepted, result: { ...accepted.result, valid: false, systemverilog: null } });
  await act(async () => { root.render(<HostedJobResult document={input} jobId="sj_1234567890abcdef" busy={true}
    onDownload={() => { throw new Error("Disabled download fired"); }} />); });
  expect(host.textContent).toContain("Graph structure: invalid");
  expect(host.querySelectorAll("button")).toHaveLength(2);
  for (const button of host.querySelectorAll("button")) { expect(button.disabled).toBe(true); button.click(); }
});

it.each([null, [], 3, { voltage: [0, 1], seed: "18446744073709551615" }].map((value) => ({ value })))(
  "does not invent compiler verdicts for analysis result $value", ({ value }) => {
    expect(hostedCompilerView(document(value))).toBeNull();
  },
);

it.each([
  { accepted: "true" }, { accepted: true }, { accepted: false, result: {}, refusal: {} },
  { accepted: false, result: null, refusal: null }, { ...accepted, refusal: {} },
  { ...accepted, result: null }, { ...accepted, result: {} },
  ...["ir_text", "valid", "source_replay", "systemverilog"].map((key) => ({
    ...accepted, result: { ...accepted.result, [key]: 12 },
  })),
  { ...accepted, result: { ...accepted.result, ir_text: "" } },
  { ...accepted, result: { ...accepted.result, source_replay: "scientifically-perfect" } },
])("refuses malformed compiler outcome %j instead of showing successful output", (value) => {
  expect(() => hostedCompilerView(document(value))).toThrow("invalid");
});

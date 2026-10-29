// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — digest-bound hosted job results

import type { HostJson } from "./hostApiTypes";
import { HostApiRefusal } from "./hostApi";

/** Actual compiler fields extracted for display, never inferred from job completion. */
interface CompilerView {
  accepted: boolean;
  irText: string | null;
  systemverilog: string | null;
  valid: boolean | null;
  sourceReplay: string | null;
}

/**
 * Read only compiler boolean/string fields from a verified result document.
 * Scientific numeric tokens stay in the original document, without rewriting.
 *
 * @param document - Exact digest-verified JSON.
 * @returns Compiler outcome when present, otherwise an analysis result.
 */
export function hostedCompilerView(document: HostJson): CompilerView | null {
  const value: unknown = JSON.parse(document.json);
  if (typeof value !== "object" || value === null || !("accepted" in value)) return null;
  if (typeof value.accepted !== "boolean" || !("result" in value) || !("refusal" in value)) {
    throw new HostApiRefusal("Compiler outcome is invalid.");
  }
  const result: unknown = value.result;
  if (!value.accepted) {
    if (result !== null || typeof value.refusal !== "object" || value.refusal === null) {
      throw new HostApiRefusal("Compiler refusal outcome is invalid.");
    }
    return { accepted: false, irText: null, systemverilog: null, valid: null, sourceReplay: null };
  }
  if (value.refusal !== null || typeof result !== "object" || result === null
    || !("ir_text" in result) || typeof result.ir_text !== "string" || result.ir_text.length === 0
    || !("valid" in result) || typeof result.valid !== "boolean"
    || !("source_replay" in result) || typeof result.source_replay !== "string"
    || !["passed", "unavailable", "not_requested"].includes(result.source_replay)
    || !("systemverilog" in result) || (result.systemverilog !== null && typeof result.systemverilog !== "string")) {
    throw new HostApiRefusal("Compiler result outcome is invalid.");
  }
  return { accepted: true, irText: result.ir_text, systemverilog: result.systemverilog,
    valid: result.valid, sourceReplay: result.source_replay };
}

/** Properties for a verified completed job result. */
interface Props {
  document: HostJson;
  jobId: string;
  busy: boolean;
  onDownload: (selection: { digest: string } | { text: string }) => void;
}
/**
 * Display exact scientific results and distinguish an explicit compiler refusal.
 *
 * @returns Result text, content binding and downloads from accepted compiler fields.
 */
export default function HostedJobResult({ document, jobId, busy, onDownload }: Props) {
  const compiler = hostedCompilerView(document);
  const irText = compiler?.irText, systemverilog = compiler?.systemverilog;
  return <section aria-label="Scientific result">
    <h3>Result for {jobId}</h3>
    {compiler !== null && <p role="status">{compiler.accepted
      ? `Compiler completed the operation. Graph structure: ${compiler.valid ? "verified" : "invalid"}. Source replay: ${compiler.sourceReplay}. Hardware qualification is separate.`
      : "Compiler refused the operation. A completed job does not mean compilation succeeded."}</p>}
    <p>{document.bytes} bytes · <code>{document.digest}</code></p>
    <button type="button" disabled={busy} onClick={() => { onDownload({ digest: document.digest }); }}>Download result JSON</button>
    {irText && <button type="button" disabled={busy} onClick={() => {
      onDownload({ text: irText });
    }}>Download native graph</button>}
    {systemverilog && <button type="button" disabled={busy} onClick={() => {
      onDownload({ text: systemverilog });
    }}>Download SystemVerilog</button>}
    <details open><summary>Scientific result JSON</summary>
      <pre style={{ maxHeight: 420, overflow: "auto", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{document.json}</pre>
    </details>
  </section>;
}

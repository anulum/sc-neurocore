// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate model workbench

/**
 * Author, check and review a candidate model package.
 *
 * The draft is the workspace's own and is saved with it exactly as typed. The
 * actions send it to the server, which validates it field by field, diffs it
 * against its parent, simulates it under its own profile and runs its
 * reference tests into a review packet. Nothing here lists the candidate in
 * the catalogue or writes a canonical file; the panel says so above the draft.
 */

import { useRef, useState, type CSSProperties } from "react";

import {
  candidateReviewPacket,
  diffCandidate,
  simulateCandidate,
  validateCandidate,
  type CandidateDiff,
  type CandidateReviewPacket,
  type CandidateRun,
  type CandidateValidation,
} from "../api/candidatesApi";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import {
  candidateDiffLines,
  candidateDiffNotice,
  candidateFileName,
  candidateRefusal,
  parseCandidateText,
} from "../candidateWorkbench";
import { useStudioStore } from "../stores/studio";

const button: CSSProperties = {
  background: "transparent",
  border: "1px solid var(--control-border)",
  borderRadius: 3,
  color: "var(--text-secondary)",
  cursor: "pointer",
  fontSize: 10,
  padding: "2px 8px",
};

const cell: CSSProperties = {
  border: "1px solid var(--border)",
  fontSize: 10,
  padding: "2px 6px",
  textAlign: "left",
  verticalAlign: "top",
};

/** What the last action produced, shown until the next one. */
interface Outcome {
  validation: CandidateValidation | null;
  diff: CandidateDiff | null;
  run: CandidateRun | null;
  packet: CandidateReviewPacket | null;
  message: string | null;
}

const NOTHING: Outcome = { validation: null, diff: null, run: null, packet: null, message: null };

/**
 * The candidate workbench panel.
 *
 * @returns The panel.
 */
export default function CandidatePanel() {
  const { candidates, setCandidateDraft } = useStudioStore();
  const draft = candidates[0]?.text ?? "";
  const [outcome, setOutcome] = useState<Outcome>(NOTHING);
  const [busy, setBusy] = useState(false);
  const [current, setCurrent] = useState("0");
  const [steps, setSteps] = useState("1000");
  const fileInput = useRef<HTMLInputElement>(null);

  /**
   * Parse the draft and run one server action on it, reporting what happens.
   *
   * @param action - What to do with the parsed candidate.
   */
  async function act(action: (document: unknown) => Promise<Partial<Outcome>>): Promise<void> {
    const parsed = parseCandidateText(draft);
    if (!parsed.ok) {
      setOutcome({ ...NOTHING, message: parsed.message });
      return;
    }
    setBusy(true);
    try {
      setOutcome({ ...NOTHING, ...(await action(parsed.document)) });
    } catch (error) {
      const refused = candidateRefusal(error);
      setOutcome(
        refused !== null
          ? { ...NOTHING, validation: refused, message: "The candidate is not valid; fix the fields listed." }
          : { ...NOTHING, message: error instanceof Error ? error.message : String(error) },
      );
    } finally {
      setBusy(false);
    }
  }

  const validation = outcome.validation;
  const diffLines = outcome.diff === null ? [] : candidateDiffLines(outcome.diff);
  const diffNotice = outcome.diff === null ? null : candidateDiffNotice(outcome.diff);

  return (
    <section
      aria-label="Candidate model"
      style={{ flex: 1, overflow: "auto", padding: "8px 12px", display: "flex", flexDirection: "column", gap: 8 }}
    >
      <h2 style={{ fontSize: 13, margin: 0 }}>Candidate model</h2>
      <p style={{ fontSize: 10, color: "var(--text-secondary)", margin: 0 }}>
        A candidate is a proposal. It is never listed as a catalogue model and nothing here changes
        a canonical file; promotion into the catalogue is a separate, reviewed step.
      </p>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        <button type="button" style={button} onClick={() => { fileInput.current?.click(); }}>
          Import candidate
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          aria-label="Import candidate package file"
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file === undefined) return;
            void file.text().then((text) => {
              setCandidateDraft(text);
              setOutcome(NOTHING);
            });
          }}
        />
        <button
          type="button"
          style={button}
          disabled={draft === ""}
          onClick={() => {
            const parsed = parseCandidateText(draft);
            downloadBrowserArtefact(
              new Blob([draft], { type: "application/json" }),
              candidateFileName(parsed.ok ? parsed.document : null, "candidate"),
            );
          }}
        >
          Export candidate
        </button>
        <button type="button" style={button} disabled={busy} onClick={() => {
          void act(async (document) => ({ validation: await validateCandidate(document) }));
        }}>Validate</button>
        <button type="button" style={button} disabled={busy} onClick={() => {
          void act(async (document) => ({ diff: await diffCandidate(document) }));
        }}>Diff against parent</button>
        <button type="button" style={button} disabled={busy} onClick={() => {
          void act(async (document) => ({ packet: await candidateReviewPacket(document) }));
        }}>Run reference tests</button>
        <button
          type="button"
          style={button}
          disabled={outcome.packet === null}
          onClick={() => {
            if (outcome.packet === null) return;
            const parsed = parseCandidateText(draft);
            downloadBrowserArtefact(
              new Blob([JSON.stringify(outcome.packet, null, 2)], { type: "application/json" }),
              candidateFileName(parsed.ok ? parsed.document : null, "review"),
            );
          }}
        >
          Export review packet
        </button>
      </div>

      <label htmlFor="candidate-draft" style={{ fontSize: 10 }}>
        Candidate package (JSON)
      </label>
      <textarea
        id="candidate-draft"
        value={draft}
        onChange={(event) => { setCandidateDraft(event.target.value); }}
        aria-describedby="candidate-outcome"
        spellCheck={false}
        rows={16}
        style={{ fontFamily: "var(--font-mono)", fontSize: 10, width: "100%", resize: "vertical" }}
      />

      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", fontSize: 10 }}>
        <label htmlFor="candidate-current">Current</label>
        <input id="candidate-current" type="number" value={current}
          onChange={(event) => { setCurrent(event.target.value); }} style={{ width: 80 }} />
        <label htmlFor="candidate-steps">Steps</label>
        <input id="candidate-steps" type="number" min={1} value={steps}
          onChange={(event) => { setSteps(event.target.value); }} style={{ width: 80 }} />
        <button type="button" style={button} disabled={busy} onClick={() => {
          void act(async (document) => ({
            run: await simulateCandidate(document, Number(current), Number(steps)),
          }));
        }}>Simulate</button>
      </div>

      <div id="candidate-outcome" role="status" style={{ fontSize: 10 }}>
        {outcome.message !== null && <p style={{ margin: 0 }}>{outcome.message}</p>}
        {validation !== null && (validation.valid ? (
          <p style={{ margin: 0 }}>
            Valid. Digest {validation.candidate_sha256?.slice(0, 16)}…
          </p>
        ) : (
          <ul aria-label="Candidate problems" style={{ margin: 0, paddingLeft: 16 }}>
            {validation.diagnostics.map((diagnostic) => (
              <li key={`${diagnostic.location}:${diagnostic.message}`}>
                <code>{diagnostic.location === "" ? "(document)" : diagnostic.location}</code>{" "}
                {diagnostic.message}
              </li>
            ))}
          </ul>
        ))}
        {outcome.run !== null && (
          <p style={{ margin: 0 }}>
            {outcome.run.spike_count} spikes in {outcome.run.steps} steps
            ({outcome.run.profile.method}, dt {outcome.run.profile.dt} {outcome.run.profile.time_unit}
            {" "}at {outcome.run.current} {outcome.run.units.current})
            {outcome.run.diverged_at_step !== null
              ? `; diverged at step ${outcome.run.diverged_at_step}: ${outcome.run.divergence ?? ""}`
              : `; final state ${Object.entries(outcome.run.final_state ?? {})
                .map(([name, value]) => `${name}=${value.toPrecision(6)}`).join(", ")}`}
          </p>
        )}
        {diffNotice !== null && <p style={{ margin: 0 }}>{diffNotice}</p>}
        {outcome.diff !== null && diffNotice === null && (
          diffLines.length === 0 ? (
            <p style={{ margin: 0 }}>The candidate changes nothing against {outcome.diff.parent}.</p>
          ) : (
            <table style={{ borderCollapse: "collapse" }}>
              <caption style={{ captionSide: "top", textAlign: "left" }}>
                What the candidate changes against {outcome.diff.parent}
              </caption>
              <thead>
                <tr>
                  {["Section", "Name", "Status", "Detail"].map((column) => (
                    <th key={column} scope="col" style={cell}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {diffLines.map((line) => (
                  <tr key={`${line.section}:${line.name}`}>
                    <td style={cell}>{line.section}</td>
                    <th scope="row" style={cell}>{line.name}</th>
                    <td style={cell}>{line.status}</td>
                    <td style={cell}>{line.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        )}
        {outcome.packet !== null && (
          <div>
            <p style={{ margin: 0 }}>
              Reference tests {outcome.packet.reference_tests_passed ? "passed" : "failed"}. Packet
              {" "}{outcome.packet.packet_sha256.slice(0, 16)}…
            </p>
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {outcome.packet.reference_tests.map((test) => (
                <li key={test.name}>
                  {test.name}: {test.passed ? "passed" : "failed"}
                  {test.checks.map((check) => ` · ${check.quantity} ${String(check.observed)}${check.held ? "" : " (outside bounds)"}`).join("")}
                </li>
              ))}
            </ul>
            <p style={{ margin: 0, color: "var(--text-secondary)" }}>Not established:</p>
            <ul style={{ margin: 0, paddingLeft: 16, color: "var(--text-secondary)" }}>
              {outcome.packet.not_established.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

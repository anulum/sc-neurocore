// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Explicit catalogue-to-candidate draft adoption

import { useEffect, useRef, useState } from "react";
import { candidateFromCatalogue } from "../api/candidatesApi";
import type { CatalogueCandidate } from "../candidateCatalogue";

/** Selected catalogue identity and the workspace's existing draft custody. */
interface Props {
  modelName: string;
  draft: string;
  onAdopt: (text: string) => void;
}

/**
 * Preview a complete canonical schema before explicitly replacing a workspace draft.
 *
 * @param props - Selected identity, current draft and its owned update action.
 * @returns Source provenance, an untouched preview and the explicit adoption action.
 */
export default function CandidateCatalogueStart({ modelName, draft, onAdopt }: Props) {
  const [proposal, setProposal] = useState<CatalogueCandidate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sequence = useRef(0);
  useEffect(() => {
    sequence.current += 1;
    setProposal(null); setError(null); setBusy(false);
    return () => { sequence.current += 1; };
  }, [modelName]);

  /** Fetch the selected model, retaining newer selection and unmounted-view ownership. */
  async function prepare(): Promise<void> {
    const request = ++sequence.current;
    setBusy(true); setError(null); setProposal(null);
    try {
      const response = await candidateFromCatalogue(modelName);
      if (request === sequence.current) setProposal(response);
    } catch (reason) {
      if (request === sequence.current) setError(String(reason));
    } finally {
      if (request === sequence.current) setBusy(false);
    }
  }

  return <section aria-label="Start candidate from catalogue">
    <p>Selected catalogue model: {modelName || "none"}</p>
    <button type="button" disabled={busy || modelName === ""} onClick={() => { void prepare(); }}>
      {busy ? "Preparing catalogue draft…" : "Prepare catalogue draft"}
    </button>
    {error !== null && <p role="alert">{error}</p>}
    {proposal !== null && proposal.parent_model === modelName && <>
      <p>Canonical schema {proposal.parent_schema}; SHA-256 {proposal.parent_schema_sha256}.</p>
      <p>Complete the candidate authors, reference tests, assumptions and any blank units before
        validation. Preparing this proposal does not change the current candidate or catalogue.</p>
      <textarea aria-label="Catalogue candidate preview" readOnly rows={8}
        value={proposal.candidate_text} style={{ width: "100%" }} />
      <button type="button" onClick={() => {
        onAdopt(proposal.candidate_text); setProposal(null);
      }}>{draft === "" ? "Use catalogue draft" : "Replace current candidate draft"}</button>
    </>}
  </section>;
}

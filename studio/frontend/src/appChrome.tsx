// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The small shared widgets the Studio shell is built from.
 *
 * They live here so the composition root owns layout and nothing else. Each is
 * presentational: none reads the store, and every one of them is told what to
 * show and what to do when clicked.
 */

import type { PanelCapabilityState } from "./capabilityShell";

/**
 * One action button, in one of three weights.
 *
 * The primary weight (filled with the accent) is for the one action a view is
 * built around, such as running the simulation; the outline weight is for the
 * other actions that compute something; the ghost weight is for import,
 * export and reset. The header used to paint every action its own colour,
 * which made eleven equally loud buttons with no order among them.
 *
 * @param props - The button's label, what to do when it is clicked, its
 *   weight (`outline` or `ghost`, primary otherwise), and its optional title,
 *   disabled state and test handle.
 * @returns The button.
 */
export function Btn({
  label,
  onClick,
  disabled,
  outline,
  ghost,
  title,
  testId,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  outline?: boolean;
  ghost?: boolean;
  title?: string;
  /** Stable handle for browser tests; several buttons share a label. */
  testId?: string;
}) {
  const weight = ghost === true ? "btn--ghost" : outline === true ? "btn--outline" : "btn--primary";
  return (
    <button
      type="button"
      className={`btn-simulate btn ${weight}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-testid={testId}
    >
      {label}
    </button>
  );
}

/**
 * What a panel shows instead of itself when its capability is unavailable.
 *
 * It names the capability, its status, the server's message and the
 * requirements that were not met -- rather than an empty panel, which reads as
 * a broken build rather than a deployment that is missing something.
 *
 * @param props - The panel's capability state.
 * @returns The blocked panel.
 */
export function CapabilityUnavailable({ state }: { state: PanelCapabilityState }) {
  return (
    <div className="capability-blocked-panel">
      <div className="capability-blocked-title">{state.title}</div>
      <div className="capability-blocked-status">{state.status}</div>
      <p>{state.message}</p>
      {state.requirements.length > 0 && (
        <ul>
          {state.requirements.map((requirement) => (
            <li key={requirement}>{requirement}</li>
          ))}
        </ul>
      )}
      <div className="capability-blocked-meta">
        {state.evidence.length > 0 && <span>Evidence: {state.evidence.join(", ")}</span>}
        {state.docsPath && (
          <a href={`/${state.docsPath}`} target="_blank" rel="noreferrer">
            Documentation
          </a>
        )}
      </div>
    </div>
  );
}

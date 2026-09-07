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
 * Pick a button's background, falling back to the accent colour.
 *
 * A function rather than an inline `||` so the reason has somewhere to live:
 * callers pass `""` for "no particular colour", and `??` would render a button
 * with no background at all.
 *
 * @param colour - The colour the caller asked for, if any.
 * @returns The colour to paint.
 */
function presentColour(colour: string | undefined): string {
  return colour !== undefined && colour.length > 0 ? colour : "var(--accent)";
}

/**
 * One tab in the Studio's panel bar.
 *
 * @param props - The tab's label, its colour when active, whether it is
 *   active, whether it is disabled, its title, and what to do when it is
 *   clicked.
 * @returns The tab button.
 */
export function Tab({
  active,
  color,
  label,
  onClick,
  disabled,
  title,
}: {
  active: boolean;
  color: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        padding: "2px 6px",
        fontSize: 9,
        fontWeight: 600,
        fontFamily: "var(--font-ui)",
        lineHeight: 1.4,
        background: active ? color : "transparent",
        color: active ? "var(--bg-primary)" : disabled ? "var(--text-muted)" : "var(--text-secondary)",
        border: "1px solid var(--control-border)",
        cursor: disabled ? "not-allowed" : "pointer",
        whiteSpace: "nowrap",
        opacity: disabled ? 0.45 : 1,
      }}
    >
      {label}
    </button>
  );
}

/**
 * One small action button.
 *
 * @param props - The button's label, what to do when it is clicked, and its
 *   optional colour, outline, title, disabled state and test handle.
 * @returns The button.
 */
export function Btn({
  label,
  onClick,
  disabled,
  color,
  outline,
  title,
  testId,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  color?: string;
  outline?: boolean;
  title?: string;
  /** Stable handle for browser tests; several buttons share a label. */
  testId?: string;
}) {
  return (
    <button
      className="btn-simulate"
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-testid={testId}
      style={{
        background: outline ? "transparent" : presentColour(color),
        border: outline ? "1px solid var(--control-border)" : "none",
        color: outline ? "var(--text-muted)" : "var(--bg-primary)",
        padding: "2px 7px",
        fontSize: 10,
      }}
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

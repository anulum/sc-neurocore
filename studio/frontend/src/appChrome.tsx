// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Shared chrome widgets for the Studio application shell.
 *
 * Extracted from App so the composition root only owns layout orchestration
 * while tab/button/capability blocked-panel presentation stays cohesive.
 */

import type { PanelCapabilityState } from "./capabilityShell";

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
        border: "1px solid var(--border)",
        cursor: disabled ? "not-allowed" : "pointer",
        whiteSpace: "nowrap",
        opacity: disabled ? 0.45 : 1,
      }}
    >
      {label}
    </button>
  );
}

export function Btn({
  label,
  onClick,
  disabled,
  color,
  outline,
  title,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  color?: string;
  outline?: boolean;
  title?: string;
}) {
  return (
    <button
      className="btn-simulate"
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        background: outline ? "transparent" : color || "var(--accent)",
        border: outline ? "1px solid var(--border)" : "none",
        color: outline ? "var(--text-muted)" : "var(--bg-primary)",
        padding: "2px 7px",
        fontSize: 10,
      }}
    >
      {label}
    </button>
  );
}

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

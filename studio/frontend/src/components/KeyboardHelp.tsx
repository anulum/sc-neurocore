// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useState, useEffect } from "react";

const SHORTCUTS = [
  ["Space", "Run simulation"],
  ["1", "Trace view"],
  ["2", "Phase portrait"],
  ["3", "f-I curve"],
  ["4", "Bifurcation"],
  ["5", "Sensitivity"],
  ["Scroll", "Zoom trace (time axis)"],
  ["Drag", "Pan trace"],
  ["Dbl-click", "Reset zoom"],
  ["?", "Toggle this help"],
];

export default function KeyboardHelp() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "?" || e.key === "/") {
        e.preventDefault();
        setShow((s) => !s);
      }
      if (e.key === "Escape") setShow(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (!show) return null;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "rgba(0,0,0,0.6)",
    }} onClick={() => setShow(false)}>
      <div style={{
        background: "var(--bg-secondary)", border: "1px solid var(--border)",
        borderRadius: 8, padding: "16px 24px", minWidth: 260,
      }} onClick={(e) => e.stopPropagation()}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "var(--text-primary)" }}>
          Keyboard Shortcuts
        </div>
        {SHORTCUTS.map(([key, desc]) => (
          <div key={key} style={{
            display: "flex", justifyContent: "space-between", gap: 16,
            padding: "3px 0", fontSize: 11,
          }}>
            <kbd style={{
              background: "var(--bg-tertiary)", border: "1px solid var(--border)",
              borderRadius: 3, padding: "1px 6px", fontFamily: "var(--font-mono)",
              fontSize: 10, color: "var(--accent)",
            }}>{key}</kbd>
            <span style={{ color: "var(--text-secondary)" }}>{desc}</span>
          </div>
        ))}
        <div style={{ marginTop: 8, fontSize: 9, color: "var(--text-muted)", textAlign: "center" }}>
          Press ? or Esc to close
        </div>
      </div>
    </div>
  );
}

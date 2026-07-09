// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { CSSProperties } from "react";

export interface EvidenceSummaryItem {
  label: string;
  value: string;
}

export type EvidenceSummaryVariant = "banner" | "grid" | "overlay" | "panel";

function buildContainerStyle(variant: EvidenceSummaryVariant): CSSProperties {
  if (variant === "banner") {
    return {
      padding: "6px 12px",
      borderBottom: "1px solid var(--border)",
      background: "var(--bg-primary)",
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
      gap: 6,
      fontSize: 9,
      color: "var(--text-muted)",
    };
  }
  if (variant === "grid") {
    return {
      marginTop: 6,
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))",
      gap: 6,
      color: "var(--text-muted)",
    };
  }
  if (variant === "overlay") {
    return {
      position: "absolute",
      top: 8,
      right: 8,
      zIndex: 2,
      display: "flex",
      gap: 10,
      flexWrap: "wrap",
      maxWidth: "calc(100% - 16px)",
      padding: "4px 8px",
      border: "1px solid var(--border)",
      borderRadius: 4,
      background: "rgba(13,17,23,0.92)",
      color: "var(--text-secondary)",
      fontFamily: "var(--font-mono)",
      fontSize: 10,
      pointerEvents: "none",
    };
  }
  return {
    marginTop: 4,
    padding: 4,
    border: "1px solid var(--border)",
    borderRadius: 4,
    color: "var(--text-muted)",
    fontSize: 9,
    lineHeight: 1.5,
  };
}

function renderItem(item: EvidenceSummaryItem, variant: EvidenceSummaryVariant) {
  if (variant === "banner") {
    return (
      <div key={item.label}>
        <span>{item.label}</span>
        <span style={{ fontWeight: 700, marginLeft: 4 }}>{item.value}</span>
      </div>
    );
  }
  if (variant === "grid" || variant === "overlay") {
    return <span key={item.label}>{item.label} {item.value}</span>;
  }
  return <div key={item.label}>{item.label} {item.value}</div>;
}

export default function EvidenceSummaryStrip({
  items,
  variant,
}: {
  items: EvidenceSummaryItem[];
  variant: EvidenceSummaryVariant;
}) {
  return (
    <div style={buildContainerStyle(variant)}>
      {items.map((item) => renderItem(item, variant))}
    </div>
  );
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

export interface TierMeta {
  short: string;
  label: string;
  color: string;
  show: boolean;
}

/** Describe a completeness tier + evidence kind for display.
 *
 * Tier 2 is scientifically curated (verified provenance + parameter units);
 * Tier 3 is engineering-verified (>=2 parity-checked backends + a reproducible
 * golden trace). Tier 0/1 are below the curated bar and carry no evidence badge.
 */
export function tierMeta(tier: number, evidenceKind: string): TierMeta {
  if (tier >= 3) {
    return { short: "T3", label: `engineering-verified · ${evidenceKind || "measured"}`, color: "var(--success)", show: true };
  }
  if (tier === 2) {
    return { short: "T2", label: `scientifically curated · ${evidenceKind || "curated"}`, color: "var(--accent)", show: true };
  }
  return { short: `T${tier}`, label: "declared", color: "var(--text-muted)", show: false };
}

export default function EvidenceTierBadge({
  tier,
  evidenceKind,
  full = false,
}: {
  tier: number;
  evidenceKind: string;
  full?: boolean;
}) {
  const meta = tierMeta(tier, evidenceKind);
  if (!meta.show) return null;
  return (
    <span
      title={`Tier ${tier} — ${meta.label}`}
      style={{
        fontSize: 8,
        padding: "0 4px",
        borderRadius: 2,
        fontWeight: 700,
        color: "var(--bg-primary)",
        background: meta.color,
        whiteSpace: "nowrap",
      }}
    >
      {full ? `${meta.short} · ${meta.label}` : meta.short}
    </span>
  );
}

/** Dual-axis science (S0–S5) + silicon (none / H0–H5) readiness badges. */
export function DualAxisBadge({
  scienceLabel,
  siliconLabel,
  scienceTier,
  siliconTier,
  full = false,
}: {
  scienceLabel: string;
  siliconLabel: string;
  scienceTier?: number;
  siliconTier?: number | null;
  full?: boolean;
}) {
  const sColor =
    (scienceTier ?? 0) >= 5
      ? "var(--success)"
      : (scienceTier ?? 0) >= 3
        ? "var(--accent)"
        : "var(--text-muted)";
  const hEnrolled = siliconTier !== null && siliconTier !== undefined;
  const hColor =
    hEnrolled && (siliconTier as number) >= 2
      ? "var(--success)"
      : hEnrolled
        ? "var(--accent)"
        : "var(--bg-tertiary)";
  const sTitle = `Science readiness ${scienceLabel}${scienceTier !== undefined ? ` (S${scienceTier})` : ""}`;
  const hTitle = hEnrolled
    ? `Silicon readiness ${siliconLabel} (H${siliconTier})`
    : "Silicon readiness not enrolled (none)";
  return (
    <span style={{ display: "inline-flex", gap: 2, alignItems: "center" }}>
      <span
        title={sTitle}
        style={{
          fontSize: 8,
          padding: "0 4px",
          borderRadius: 2,
          fontWeight: 700,
          color: "var(--bg-primary)",
          background: sColor,
          whiteSpace: "nowrap",
        }}
      >
        {full ? `science ${scienceLabel}` : scienceLabel}
      </span>
      <span
        title={hTitle}
        style={{
          fontSize: 8,
          padding: "0 4px",
          borderRadius: 2,
          fontWeight: 700,
          color: hEnrolled ? "var(--bg-primary)" : "var(--text-muted)",
          background: hColor,
          whiteSpace: "nowrap",
          border: hEnrolled ? "none" : "1px solid var(--border)",
        }}
      >
        {full ? `silicon ${siliconLabel}` : siliconLabel}
      </span>
    </span>
  );
}

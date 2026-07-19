// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Persistent development-preview shell banner
import {
  developmentPreviewBannerModel,
  type StudioDeploymentProfile,
} from "../developmentPreview";

export interface DevelopmentPreviewBannerProps {
  deploymentProfile: StudioDeploymentProfile;
  releaseValidated?: boolean;
}

/**
 * Persistent shell banner that labels Studio as a development preview.
 */
export default function DevelopmentPreviewBanner({
  deploymentProfile,
  releaseValidated = false,
}: DevelopmentPreviewBannerProps) {
  const model = developmentPreviewBannerModel(deploymentProfile, { releaseValidated });
  if (!model.visible) {
    return null;
  }
  return (
    <div
      data-testid="development-preview-banner"
      role="status"
      aria-label={model.label}
      style={{
        padding: "4px 12px",
        borderBottom: "1px solid var(--border)",
        background: "rgba(210, 153, 34, 0.12)",
        color: "var(--text-secondary)",
        fontSize: 11,
        fontFamily: "var(--font-ui)",
        display: "flex",
        gap: 10,
        alignItems: "baseline",
        flexWrap: "wrap",
      }}
    >
      <strong style={{ color: "var(--warning, #d29922)" }}>{model.label}</strong>
      <span>{model.detail}</span>
    </div>
  );
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio readiness panel
import type { StudioReadinessModel } from "../studioReadiness";

export interface StudioReadinessPanelProps {
  model: StudioReadinessModel;
  onOpenAdmin: () => void;
  onRefresh: () => void;
  primaryActionLabel?: string;
}

/** Render the promotion-readiness checklist derived from operator status. */
export default function StudioReadinessPanel({
  model,
  onOpenAdmin,
  onRefresh,
  primaryActionLabel = "Open admin",
}: StudioReadinessPanelProps) {
  return (
    <div className={`studio-readiness studio-readiness-${model.posture}`}>
      <div className="studio-readiness-heading">
        <div>
          <div className="studio-readiness-title">Readiness</div>
          <div className="studio-readiness-headline">{model.headline}</div>
          <div className="studio-readiness-subhead">{model.subhead}</div>
        </div>
        <span>{model.blockingCount} blockers / {model.warningCount} warnings</span>
      </div>
      <div className="studio-readiness-list">
        {model.items.map((item) => (
          <div key={item.key} className="studio-readiness-row">
            <span className={`studio-readiness-status-${item.status}`}>{item.status}</span>
            <strong>{item.label}</strong>
            <small>{item.value}</small>
            <em>{item.action}</em>
          </div>
        ))}
      </div>
      <div className="studio-readiness-actions">
        <button onClick={onRefresh} type="button">Refresh</button>
        <button onClick={onOpenAdmin} type="button">{primaryActionLabel}</button>
      </div>
    </div>
  );
}

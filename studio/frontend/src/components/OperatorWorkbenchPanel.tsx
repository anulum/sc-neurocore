// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator workbench panel
import type {
  OperatorWorkbenchCardKey,
  OperatorWorkbenchCardStatus,
  OperatorWorkbenchState,
} from "../operatorWorkbenchState";

export interface OperatorWorkbenchPanelProps {
  onExportEvidence: () => void;
  onOpenAdmin: () => void;
  onOpenCompiler: () => void;
  onOpenProjects: () => void;
  onRunSimulation: () => void;
  state: OperatorWorkbenchState;
}

const STATUS_LABELS: Record<OperatorWorkbenchCardStatus, string> = {
  active: "active",
  blocked: "blocked",
  ready: "ready",
  warning: "check",
};

const STATUS_CLASS: Record<OperatorWorkbenchCardStatus, string> = {
  active: "operator-workbench-status-active",
  blocked: "operator-workbench-status-blocked",
  ready: "operator-workbench-status-ready",
  warning: "operator-workbench-status-warning",
};

export default function OperatorWorkbenchPanel({
  onExportEvidence,
  onOpenAdmin,
  onOpenCompiler,
  onOpenProjects,
  onRunSimulation,
  state,
}: OperatorWorkbenchPanelProps) {
  return (
    <section className="operator-workbench" aria-label="Operator workbench">
      <div className="operator-workbench-heading">
        <div>
          <div className="operator-workbench-title">Operator workbench</div>
          <div className="operator-workbench-subtitle">{state.headline}</div>
        </div>
        <span aria-label="Operator workbench progress">{state.subhead}</span>
      </div>
      <div className="operator-workbench-grid">
        {state.cards.map((card) => (
          <article className="operator-workbench-card" key={card.key} data-card={card.key}>
            <div className="operator-workbench-card-top">
              <span>{card.title}</span>
              <span className={STATUS_CLASS[card.status]}>{STATUS_LABELS[card.status]}</span>
            </div>
            <div className="operator-workbench-value" title={card.value}>{card.value}</div>
            <div className="operator-workbench-detail" title={card.detail}>{card.detail}</div>
            <button
              disabled={actionDisabled(card.key, state)}
              onClick={() => runAction(card.key, {
                exportEvidence: onExportEvidence,
                openAdmin: onOpenAdmin,
                openCompiler: onOpenCompiler,
                openProjects: onOpenProjects,
                runSimulation: onRunSimulation,
              })}
              type="button"
            >
              {card.action}
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}

interface OperatorWorkbenchActions {
  exportEvidence: () => void;
  openAdmin: () => void;
  openCompiler: () => void;
  openProjects: () => void;
  runSimulation: () => void;
}

function actionDisabled(key: OperatorWorkbenchCardKey, state: OperatorWorkbenchState): boolean {
  return key === "export" && !state.evidenceActionEnabled;
}

function runAction(key: OperatorWorkbenchCardKey, actions: OperatorWorkbenchActions): void {
  switch (key) {
    case "compile":
      actions.openCompiler();
      return;
    case "evidence":
      actions.openAdmin();
      return;
    case "export":
      actions.exportEvidence();
      return;
    case "model":
    case "workspace":
      actions.openProjects();
      return;
    case "simulation":
      actions.runSimulation();
      return;
  }
}

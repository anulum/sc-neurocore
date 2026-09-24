// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useStudioStore } from "../stores/studio";
import {
  buildPipelineEvidenceModel,
  pipelineCosimSummary,
  pipelineFailureReason,
  type PipelineEvidenceModel,
} from "../pipelineEvidence";
import { CANVAS_TINTS } from "../paletteContrast";
import {
  studioNodeChangePlan,
  studioPopulationDriveLabel,
  studioProjectionLabel,
} from "../studioGraphRequests";
import type { GraphSimResult } from "../api/client";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";
import NetworkGraphConnect from "./NetworkGraphConnect";
import NetworkGraphTable from "./NetworkGraphTable";
import PopulationEditor from "./PopulationEditor";
import ProjectionEditor from "./ProjectionEditor";

/**
 * Why a pipeline run failed, in one line.
 *
 * `||` and not `??` throughout: an empty error list joins to an empty string,
 * and an empty message is no message; both must fall through to the next
 * source rather than being shown.
 *
 * @param result - The failed run.
 * @returns The reason to show, never empty.
 */
/**
 * What one population node shows on the canvas.
 *
 * @param props - The node's data, as ReactFlow carries it.
 * @returns The node's contents.
 */
function PopulationNodeContent({ data }: { data: Record<string, unknown> }) {
  const isExc = data.neuron_type === "excitatory";
  return (
    <div style={{
      padding: "8px 12px", borderRadius: isExc ? 8 : 4,
      background: isExc ? CANVAS_TINTS.excitatoryNode : CANVAS_TINTS.inhibitoryNode,
      border: `2px solid ${isExc ? "#4fc3f7" : "#ff5252"}`,
      minWidth: 100, textAlign: "center",
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: isExc ? "#4fc3f7" : "#ff5252" }}>
        {data.label as string}
      </div>
      <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>
        {data.model as string} × {data.count as number}
      </div>
      <div style={{ fontSize: 8, color: "var(--text-muted)" }}>
        {isExc ? "excitatory" : "inhibitory"} · {data.drive as string}
      </div>
    </div>
  );
}

/**
 * What a graph run produced, per population.
 *
 * @param props - The run.
 * @returns The summary.
 */
export function GraphResultSummary({ result }: { result: GraphSimResult }) {
  const populations = result.populations ?? [];
  const rejected = result.execution?.backend.rejected ?? [];
  return (
    <div style={{
      padding: "6px 12px", borderTop: "1px solid var(--border)",
      fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-secondary)",
      display: "flex", gap: 16, flexWrap: "wrap",
    }}>
      <span>Neurons: {result.n_total}</span>
      <span>Synapses: {result.topology?.n_synapses ?? 0}</span>
      <span>Spikes: {result.n_spikes}</span>
      <span>Steps: {result.n_steps} × {result.dt} ms</span>
      {populations.map((population) => (
        <span key={population.id}>
          {population.label}: {population.n_spikes} spikes, {population.mean_rate_hz.toFixed(1)} Hz
        </span>
      ))}
      <span>backend: {result.execution?.backend.selected ?? "?"}
        {rejected.length > 0 ? ` (rejected: ${rejected.map((r) => r.name).join(", ")})` : ""}</span>
      {result.spec?.graph_sha256 && <span>graph {result.spec.graph_sha256.slice(0, 12)}</span>}
    </div>
  );
}

const nodeTypes = { population: PopulationNodeContent };

/**
 * What a pipeline run is, stated beside its outcome.
 *
 * @param props - The run's evidence model.
 * @returns The strip.
 */
export function PipelineEvidenceStrip({ evidence }: { evidence: PipelineEvidenceModel }) {
  return (
    <EvidenceSummaryStrip
      variant="grid"
      items={[
        { label: "class", value: evidence.classification },
        { label: "action", value: evidence.actionKind },
        { label: "status", value: evidence.status },
        { label: "target", value: evidence.target },
        { label: "step", value: evidence.step },
        { label: "replay", value: evidence.replayRoute },
        { label: "artifacts", value: `${evidence.resultArtifact} / ${evidence.evidenceArtifact}` },
      ]}
    />
  );
}

/**
 * The network graph: a canvas, an equivalent table, and the editors.
 *
 * @returns The panel.
 */
export default function NetworkCanvas() {
  const {
    graphPopulations, graphProjections, graphSimResult, graphErrors, graphIssues, pipelineResult,
    selectedProjectionId, selectProjection, updateProjection, validateGraphAction,
    selectedPopulationId, selectPopulation, populationModelContract, graphModels,
    selectedPopulationIds, selectPopulations, duplicateSelection, graphNotice,
    addPopulation, updatePopulation, removePopulation,
    addProjection, removeProjection,
    undoGraphEdit, redoGraphEdit, graphHistory,
    simulateGraphAction, exportGraphNIR, importGraphNIR, loadGraphModels, runPipelineAction,
    isSimulating, synthTarget,
  } = useStudioStore();

  const [tableView, setTableView] = useState(false);
  const nirFileInput = useRef<HTMLInputElement>(null);
  const editorPanel = useRef<HTMLDivElement>(null);
  // Set by the table's Edit controls: the editor they open takes the focus, so
  // a keyboard user lands in it instead of hunting for it after the table.
  const [focusEditor, setFocusEditor] = useState(false);

  useEffect(() => { void loadGraphModels(); }, [loadGraphModels]);

  // Ctrl/Cmd+Z steps back through graph edits, Ctrl/Cmd+Shift+Z forward. A key
  // pressed inside a field belongs to that field, not to the graph.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") {
        return;
      }
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target?.isContentEditable) {
        return;
      }
      event.preventDefault();
      if (event.shiftKey) {
        redoGraphEdit();
      } else {
        undoGraphEdit();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => { window.removeEventListener("keydown", onKeyDown); };
  }, [redoGraphEdit, undoGraphEdit]);

  const nodes: Node[] = useMemo(() =>
    graphPopulations.map((p) => ({
      id: p.id,
      type: "population",
      position: p.position,
      data: {
        label: p.label, model: p.model, count: p.count, neuron_type: p.neuron_type,
        drive: studioPopulationDriveLabel(p.drive),
      },
    })),
    [graphPopulations],
  );

  const edges: Edge[] = useMemo(() =>
    graphProjections.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: studioProjectionLabel(e),
      style: { stroke: "var(--text-muted)", strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "var(--text-muted)" },
      labelStyle: { fontSize: 8, fill: "var(--text-muted)" },
    })),
    [graphProjections],
  );

  const onNodesChange: OnNodesChange = useCallback((changes) => {
    const plan = studioNodeChangePlan(changes, applyNodeChanges(changes, nodes), graphPopulations);
    for (const move of plan.moved) {
      updatePopulation(move.id, { position: move.position });
    }
    // A removal used to be computed and dropped, so the population and every
    // projection touching it survived and reappeared on the next render.
    for (const id of plan.removed) {
      removePopulation(id);
    }
  }, [nodes, graphPopulations, removePopulation, updatePopulation]);

  const onEdgesChange: OnEdgesChange = useCallback((changes) => {
    // The store holds the graph, so the edges ReactFlow would compute here are
    // discarded; only the removals are acted on. `applyEdgeChanges` used to be
    // called and its result voided, which read as if it mattered.
    const removedIds = new Set(
      changes.filter((c) => c.type === "remove").map((c) => c.id)
    );
    for (const id of removedIds) removeProjection(id);
  }, [edges, removeProjection]);

  // A click on an edge opens the editor for it; a click on the empty canvas
  // closes it, because an editor for nothing is a panel with stale numbers.
  const onEdgeClick = useCallback(
    (_event: unknown, edge: { id: string }) => { selectProjection(edge.id); },
    [selectProjection],
  );
  // The canvas's own selection drives the group operations; the editor's
  // single selection is separate, because an editor edits exactly one thing.
  const onSelectionChange = useCallback(
    ({ nodes }: { nodes: { id: string }[] }) => { selectPopulations(nodes.map((node) => node.id)); },
    [selectPopulations],
  );

  const onNodeClick = useCallback(
    (_event: unknown, node: { id: string }) => { selectPopulation(node.id); },
    [selectPopulation],
  );
  const onPaneClick = useCallback(() => {
    selectProjection(null);
    selectPopulation(null);
  }, [selectPopulation, selectProjection]);

  const selectedProjection = graphProjections.find(
    (projection) => projection.id === selectedProjectionId,
  );
  const selectedPopulation = graphPopulations.find(
    (population) => population.id === selectedPopulationId,
  );

  useEffect(() => {
    if (!focusEditor) return;
    editorPanel.current?.querySelector<HTMLElement>("input, select, textarea")?.focus();
    setFocusEditor(false);
  }, [focusEditor, selectedPopulationId, selectedProjectionId]);

  const editPopulationFromTable = useCallback((id: string) => {
    selectPopulation(id);
    setFocusEditor(true);
  }, [selectPopulation]);

  const editProjectionFromTable = useCallback((id: string) => {
    selectProjection(id);
    setFocusEditor(true);
  }, [selectProjection]);

  const onConnect: OnConnect = useCallback((conn) => {
    if (conn.source && conn.target) {
      void addProjection(conn.source, conn.target);
    }
  }, [addProjection]);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      {/* Toolbar */}
      <div style={{
        padding: "6px 12px", background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border)",
        display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
          Network Canvas
        </span>
        <button onClick={() => { void addPopulation("excitatory"); }} style={{
          background: CANVAS_TINTS.excitatoryButton, color: "#4fc3f7", border: "1px solid #4fc3f7",
          padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
        }}>+ Exc</button>
        <button onClick={() => { void addPopulation("inhibitory"); }} style={{
          background: CANVAS_TINTS.inhibitoryButton, color: "#ff5252", border: "1px solid #ff5252",
          padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
        }}>+ Inh</button>
        <button
          onClick={undoGraphEdit}
          disabled={graphHistory.past.length === 0}
          aria-label="Undo the last graph edit"
          title="Undo the last graph edit (Ctrl+Z). Moving a node is not an edit."
          style={{
            background: "transparent", color: "var(--text-muted)", border: "1px solid var(--control-border)",
            padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
          }}
        >Undo</button>
        <button
          onClick={redoGraphEdit}
          disabled={graphHistory.future.length === 0}
          aria-label="Redo the last undone graph edit"
          title="Redo the last undone graph edit (Ctrl+Shift+Z)"
          style={{
            background: "transparent", color: "var(--text-muted)", border: "1px solid var(--control-border)",
            padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
          }}
        >Redo</button>
        <button
          onClick={() => void duplicateSelection()}
          disabled={selectedPopulationIds.length === 0}
          aria-label={
            selectedPopulationIds.length === 1
              ? "Duplicate the selected population"
              : `Duplicate the ${selectedPopulationIds.length} selected populations and the projections between them`
          }
          title="Copy the selection, with the projections whose both ends are inside it. A projection leaving the selection is not copied."
          style={{
            background: "transparent", color: "var(--text-muted)", border: "1px solid var(--control-border)",
            padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
          }}
        >Duplicate</button>
        <button onClick={() => { void simulateGraphAction(); }} disabled={isSimulating || graphPopulations.length === 0} style={{
          background: "#81c784", color: "#0d1117", border: "none",
          padding: "3px 10px", fontSize: 10, cursor: "pointer",
        }}>
          {isSimulating ? "..." : "Simulate"}
        </button>
        <button onClick={() => { void runPipelineAction(); }} disabled={isSimulating || graphPopulations.length === 0} style={{
          background: "#a5d6a7", color: "#0d1117", border: "none",
          padding: "3px 10px", fontSize: 10, cursor: "pointer",
        }}>
          Pipeline → {synthTarget.toUpperCase()}
        </button>
        <button
          onClick={() => { setTableView((shown) => !shown); }}
          aria-pressed={tableView}
          title="Everything the canvas shows, as a table a screen reader can read"
          style={{
            background: tableView ? "var(--border)" : "transparent",
            color: tableView ? "var(--text)" : "var(--text-muted)",
            border: "1px solid var(--control-border)",
            padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
          }}
        >Table view</button>
        <button onClick={() => { void exportGraphNIR(); }} disabled={graphPopulations.length === 0} style={{
          background: "transparent", color: "var(--text-muted)", border: "1px solid var(--control-border)",
          padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
        }}>Export NIR</button>
        <button
          onClick={() => { nirFileInput.current?.click(); }}
          title="Replace the canvas with the network in a NIR file, or in a graph envelope saved as .json"
          style={{
            background: "transparent", color: "var(--text-muted)", border: "1px solid var(--control-border)",
            padding: "2px 8px", fontSize: 10, cursor: "pointer", borderRadius: 3,
          }}
        >Import NIR</button>
        <input
          ref={nirFileInput}
          accept=".nir,.h5,.hdf5,application/x-hdf5,.json,application/json"
          aria-label="Import NIR network file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            void importGraphNIR(file);
            event.target.value = "";
          }}
          style={{ display: "none" }}
          type="file"
        />
        <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
          {graphPopulations.length} pop · {graphProjections.length} proj · drag to connect
        </span>
      </div>

      {graphNotice !== null && (
        <div role="status" style={{
          padding: "4px 12px", borderTop: "1px solid var(--border)",
          fontSize: 10, color: "var(--text-secondary)",
        }}>
          {graphNotice}
        </div>
      )}

      {/* Errors */}
      {graphErrors.length > 0 && (
        <div style={{
          padding: "4px 12px", background: CANVAS_TINTS.errorStrip, fontSize: 10, color: "#ff5252",
        }}>
          {graphErrors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {/* The table or the canvas, and beside either the selected object's editor */}
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
      {tableView && (
        <div style={{ flex: 1, overflow: "auto", padding: "8px 12px" }}>
          <NetworkGraphTable
            populations={graphPopulations}
            projections={graphProjections}
            issues={graphIssues}
            onRemovePopulation={removePopulation}
            onEditPopulation={editPopulationFromTable}
            onEditProjection={editProjectionFromTable}
            onRemoveProjection={removeProjection}
          />
          <NetworkGraphConnect
            populations={graphPopulations}
            onConnect={(sourceId, targetId) => { void addProjection(sourceId, targetId); }}
          />
        </div>
      )}
      <div style={{ display: tableView ? "none" : "block", flex: 1, position: "relative" }}>
        {graphPopulations.length === 0 ? (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center",
            justifyContent: "center", color: "var(--text-muted)", fontSize: 11,
          }}>
            Add excitatory and inhibitory populations, then drag between nodes to connect
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onEdgeClick={onEdgeClick}
            onNodeClick={onNodeClick}
            onSelectionChange={onSelectionChange}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            style={{ background: "var(--bg-primary)" }}
          >
            <Background color="var(--border)" gap={24} />
            <Controls position="bottom-right" />
          </ReactFlow>
        )}
      </div>
      <div ref={editorPanel} style={{ display: "contents" }}>
      {selectedPopulation !== undefined && (
        <PopulationEditor
          population={selectedPopulation}
          models={graphModels}
          contract={populationModelContract}
          issues={graphIssues}
          onChange={updatePopulation}
          onValidate={() => void validateGraphAction()}
        />
      )}
      {selectedProjection !== undefined && (
        <ProjectionEditor
          projection={selectedProjection}
          populations={graphPopulations}
          issues={graphIssues}
          onChange={updateProjection}
          onValidate={() => void validateGraphAction()}
        />
      )}
      </div>
      </div>

      {/* Pipeline result */}
      {pipelineResult && (
        <div style={{
          padding: "6px 12px", borderTop: "1px solid var(--border)",
          fontSize: 10, color: pipelineResult.success ? "var(--text-secondary)" : "#ff5252",
          background: pipelineResult.success ? CANVAS_TINTS.pipelineSucceeded : CANVAS_TINTS.pipelineFailed,
        }}>
          {pipelineResult.success
            ? `Pipeline complete: ${pipelineResult.pipeline} → ${pipelineResult.target.toUpperCase()}`
            : `Pipeline failed at ${pipelineResult.step}: ${pipelineFailureReason(pipelineResult)}`}
          {pipelineCosimSummary(pipelineResult) !== null && (
            <div style={{ marginTop: 2 }}>{pipelineCosimSummary(pipelineResult)}</div>
          )}
          <PipelineEvidenceStrip evidence={buildPipelineEvidenceModel(pipelineResult)} />
        </div>
      )}

      {/* Sim results summary */}
      {graphSimResult?.success && <GraphResultSummary result={graphSimResult} />}
    </div>
  );
}

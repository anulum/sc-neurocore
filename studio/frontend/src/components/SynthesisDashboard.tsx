import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

function ResourceBar({ label, used, total, color }: {
  label: string; used: number; total: number; color: string;
}) {
  const pct = total > 0 ? Math.min((used / total) * 100, 100) : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {used} / {total} ({pct.toFixed(1)}%)
        </span>
      </div>
      <div style={{
        height: 8, background: "var(--bg-tertiary)", borderRadius: 4, overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${pct}%`, background: color,
          borderRadius: 4, transition: "width 0.3s",
        }} />
      </div>
    </div>
  );
}

export default function SynthesisDashboard() {
  const {
    synthResult, synthTarget, toolsAvailable, svSource,
    setSynthTarget, runSynthesis, checkSynthTools, isSimulating,
  } = useStudioStore();

  useEffect(() => { checkSynthTools(); }, [checkSynthTools]);

  const targets = ["ice40", "ecp5", "gowin", "xilinx"];
  const hasSV = svSource.length > 0;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "auto" }}>
      {/* Header */}
      <div style={{
        padding: "8px 12px", background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border)",
        display: "flex", gap: 8, alignItems: "center",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
          FPGA Synthesis
        </span>
        <select
          value={synthTarget}
          onChange={(e) => setSynthTarget(e.target.value)}
          style={{ fontSize: 10, padding: "2px 6px" }}
        >
          {targets.map((t) => (
            <option key={t} value={t}>{t.toUpperCase()}</option>
          ))}
        </select>
        <button
          className="btn-simulate"
          onClick={runSynthesis}
          disabled={isSimulating || !hasSV}
          style={{
            background: "#a5d6a7", color: "#0d1117", border: "none",
            padding: "3px 10px", fontSize: 10,
          }}
        >
          {isSimulating ? "..." : "Synthesise"}
        </button>
        {!hasSV && (
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
            Generate Verilog first (RTL or SV button)
          </span>
        )}
      </div>

      {/* Tool status */}
      {toolsAvailable && (
        <div style={{
          padding: "4px 12px", fontSize: 9, display: "flex", gap: 12,
          borderBottom: "1px solid var(--border)", color: "var(--text-muted)",
        }}>
          {Object.entries(toolsAvailable).map(([name, info]) => (
            <span key={name} style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: info.available ? "#81c784" : "#616161",
              }} />
              {name}
            </span>
          ))}
        </div>
      )}

      {/* Results */}
      {synthResult && (
        <div style={{ padding: 12, flex: 1 }}>
          {!synthResult.success ? (
            <div style={{
              padding: 12, background: "rgba(255,82,82,0.1)", borderRadius: 4,
              color: "#ff5252", fontSize: 11,
            }}>
              {synthResult.error}
            </div>
          ) : (
            <>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)", marginBottom: 12 }}>
                {synthResult.target.toUpperCase()} — Synthesis Results
              </div>

              <ResourceBar
                label="LUTs" used={synthResult.resources.luts}
                total={synthResult.capacity.luts} color="#4fc3f7"
              />
              <ResourceBar
                label="Flip-Flops" used={synthResult.resources.ffs}
                total={synthResult.capacity.ffs} color="#81c784"
              />
              <ResourceBar
                label="Block RAMs" used={synthResult.resources.brams}
                total={synthResult.capacity.brams} color="#ffb74d"
              />
              <ResourceBar
                label="DSPs" used={synthResult.resources.dsps}
                total={synthResult.capacity.dsps} color="#ce93d8"
              />

              <div style={{
                marginTop: 12, padding: 8, background: "var(--bg-secondary)",
                borderRadius: 4, fontSize: 10, fontFamily: "var(--font-mono)",
                color: "var(--text-secondary)",
              }}>
                <div>Cells: {synthResult.resources.cells}</div>
                <div>Wires: {synthResult.resources.wires}</div>
                {synthResult.log_excerpt && (
                  <div style={{ marginTop: 6, color: "var(--text-muted)", fontSize: 9 }}>
                    {synthResult.log_excerpt}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {!synthResult && hasSV && (
        <div style={{
          flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--text-muted)", fontSize: 11,
        }}>
          Click Synthesise to run Yosys on your generated SystemVerilog
        </div>
      )}
    </div>
  );
}

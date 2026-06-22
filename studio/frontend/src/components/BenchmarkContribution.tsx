import { useEffect, useState } from "react";
import {
  contributeBenchmark,
  fetchDatabank,
  runBenchmark,
  type BenchmarkSubmission,
  type DatabankLeaderboard,
} from "../api/client";

/** A privacy-transparent preview of exactly what a contribution sends. */
export function contributionPreview(s: BenchmarkSubmission): string {
  return JSON.stringify(
    {
      cpu: s.environment.cpu,
      os: s.environment.os,
      python: s.environment.python,
      backends: s.backends.map((b) => ({
        backend: b.backend,
        speedup_over_python: b.speedup_over_python,
      })),
      bit_exact_all: s.parity.bit_exact_all,
      hardware_measurement_claimed: s.hardware_measurement_claimed,
    },
    null,
    1,
  );
}

export default function BenchmarkContribution() {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchmarkSubmission | null>(null);
  const [handle, setHandle] = useState("");
  const [contributed, setContributed] = useState(false);
  const [board, setBoard] = useState<DatabankLeaderboard | null>(null);

  function loadBoard() {
    void fetchDatabank().then(setBoard).catch(() => setBoard(null));
  }
  useEffect(loadBoard, []);

  function run() {
    setRunning(true);
    setContributed(false);
    setResult(null);
    void runBenchmark({ n_channels: 512, n_taps: 32, repeats: 12 })
      .then(setResult)
      .catch(() => setResult(null))
      .finally(() => setRunning(false));
  }

  function contribute() {
    if (!result) return;
    void contributeBenchmark(result, handle)
      .then(() => {
        setContributed(true);
        loadBoard();
      })
      .catch(() => setContributed(false));
  }

  const max = result ? Math.max(...result.backends.map((b) => b.speedup_over_python), 1) : 1;

  return (
    <div style={{ marginTop: 16, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          Run the same kernel on <b>your</b> machine — optionally contribute to the databank
        </div>
        <button type="button" onClick={run} disabled={running} style={{
          fontSize: 10, padding: "2px 10px", cursor: running ? "wait" : "pointer",
          background: "var(--accent)", color: "var(--bg-primary)", border: "none",
          borderRadius: "var(--radius)", fontWeight: 600,
        }}>
          {running ? "running…" : "Run on my machine"}
        </button>
      </div>

      {result && (
        <div style={{ display: "grid", gap: 3, marginBottom: 8 }}>
          {result.backends.map((b) => (
            <div key={b.backend} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
              <span style={{ width: 56, fontFamily: "var(--font-mono)" }}>{b.backend}</span>
              <div style={{ flex: 1, background: "var(--bg-tertiary)", borderRadius: 2, height: 13 }}>
                <div style={{
                  width: `${(b.speedup_over_python / max) * 100}%`, height: "100%",
                  background: "var(--success)", borderRadius: 2, minWidth: 2,
                }} />
              </div>
              <span style={{ width: 96, textAlign: "right", fontFamily: "var(--font-mono)" }}>
                {b.speedup_over_python.toFixed(1)}× · {b.median_call_ms.toFixed(3)}ms
              </span>
            </div>
          ))}
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
            {result.environment.cpu} · {result.environment.os}
            {result.parity.bit_exact_all ? " · all bit-exact" : " · ⚠ parity differs"}
          </div>
        </div>
      )}

      {result && !contributed && (
        <details style={{ fontSize: 10, color: "var(--text-secondary)" }}>
          <summary style={{ cursor: "pointer", color: "var(--accent)" }}>
            Contribute to the databank (opt-in) — review exactly what is sent
          </summary>
          <pre style={{
            maxHeight: 160, overflow: "auto", margin: "4px 0", padding: 6, fontSize: 9,
            background: "var(--bg-primary)", border: "1px solid var(--border)",
            borderRadius: "var(--radius)", whiteSpace: "pre-wrap",
          }}>{contributionPreview(result)}</pre>
          <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 4 }}>
            No hostname, user, IP or machine-id is collected. The handle is optional.
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input value={handle} onChange={(e) => setHandle(e.target.value)}
              placeholder="handle (optional)" maxLength={40} style={{
                fontSize: 10, padding: "2px 6px", flex: 1,
                background: "var(--bg-tertiary)", color: "var(--text-primary)",
                border: "1px solid var(--border)", borderRadius: "var(--radius)",
              }} />
            <button type="button" onClick={contribute} style={{
              fontSize: 10, padding: "2px 10px", cursor: "pointer",
              background: "var(--bg-secondary)", color: "var(--text-secondary)",
              border: "1px solid var(--border)", borderRadius: "var(--radius)",
            }}>Contribute</button>
          </div>
        </details>
      )}

      {contributed && (
        <div style={{ fontSize: 10, color: "var(--success)" }}>✓ contributed — thank you</div>
      )}

      {board && board.count > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 3 }}>
            Databank leaderboard ({board.count})
          </div>
          {board.entries.slice(0, 5).map((e, i) => (
            <div key={i} style={{ fontSize: 10, display: "flex", gap: 8, fontFamily: "var(--font-mono)" }}>
              <span style={{ color: "var(--success)", width: 70 }}>
                {e.fastest_backend} {e.speedup.toFixed(0)}×
              </span>
              <span style={{ color: "var(--text-muted)", flex: 1, overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                {e.cpu}{e.handle ? ` · ${e.handle}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

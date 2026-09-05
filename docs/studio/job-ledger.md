<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# The job ledger

Studio jobs are recorded in `job_ledger.sqlite3`, inside the job root, beside
the per-job sandbox directories it describes. The records survive the process
that made them, so a restarted API still knows what it ran, and a second API
process over the same root sees the same jobs.

Schema `studio.job-ledger.v1`. Single host by design: SQLite in WAL mode
serialises the writers sharing one root. A distributed worker contract is a
separate obligation and is not implied here.

## What a job carries

| Field | Meaning |
|---|---|
| `owner`, `workspace` | Who ran it and in which scope. Reads can be scoped by both |
| `request_id` | The caller's correlation id |
| `idempotency_key` | Submitting the same key twice for one actor and workspace returns the first job |
| `experiment_sha256` | The effective experiment the job executes, when it has one |
| `admission` | The admission decision recorded at submission |
| `lease_owner`, `lease_expires_at_utc`, `heartbeat_at_utc` | Which supervisor holds it, and until when without a heartbeat |
| `result`, `artifacts` | Committed in the same transaction as the terminal status |

Every status change also appends a row to `job_transitions`, with the previous
status, the new one, the clock, the actor and a reason. That log is append-only
because triggers refuse to update or delete its rows.

## States

`pending` → `running` → `completed` / `failed`, with `cancelling` → `cancelled`
and `timed_out` as the other ordinary ends. Recovery adds two:

- **`interrupted`** — the job did not finish and its supervisor is provably
  gone. Terminal.
- **`unknown`** — the job may still be running under a supervisor this host
  cannot probe. Not terminal; it awaits verification.

A job asked to cancel before its supervisor marks it running stays
`cancelling`: it did start, and it is already winding down. A terminal record
is never rewritten, so a late claim of success on an interrupted job is
refused rather than accepted.

## Recovery on startup

Constructing a `StudioJobManager` over an existing root reconciles every job
that had not finished. Pass `reconcile=False` to skip it, and call
`manager.reconcile()` explicitly instead.

| What recovery finds | What it does |
|---|---|
| Lease held by a process still running, not expired | Leaves the job alone |
| Lease held by a process on this host that is gone | `interrupted` |
| Lease expired without a heartbeat | `interrupted` |
| Lease held by a supervisor this host cannot probe | `unknown` |

A lease already stamped with this process's own identity belongs to the
incarnation that died: recovery runs at startup, before this process supervises
anything.

Nothing is ever promoted to `completed`, and no side effect is repeated to find
out what happened. A result that was never committed is not a result.

The decisions are readable afterwards:

```python
for decision in manager.last_reconciliation:
    print(decision.job_id, decision.previous_status, "->", decision.status, decision.reason)
```

`GET /api/studio/jobs/status` reports the same list under `recovery`, alongside
`interrupted_count` and `unknown_count` (payload `studio.jobs.status.v2`).

### A crash between the artifact and the outcome

A job that wrote an artifact and then died leaves the file on disk and no
manifest entry, because the manifest is committed with the terminal status in
one transaction. Recovery marks the job `interrupted` and leaves the file
alone: the evidence is preserved without being promoted into a result the job
never produced. Read it from the job directory; do not add it to a manifest by
hand.

## Migration

`schema_meta` holds the stored version. Opening a ledger runs forward
migrations up to this build's version. A ledger written by a *newer* build is
refused with `StudioJobLedgerCorrupt` rather than downgraded, because
downgrading would silently drop columns. Upgrade the package instead.

## Retention and backup

- The ledger is one file. Back it up with the job root it describes; separating
  them turns records into orphans and artifacts into anonymous bytes.
- WAL mode leaves `job_ledger.sqlite3-wal` and `-shm` beside it. Copy the
  ledger only while no Studio process is writing, or use `sqlite3 … ".backup"`,
  which is consistent under concurrent writers.
- `manager.purge_terminal_record(job_id)` deletes a terminal job's directory,
  its record and its whole transition history. An unfinished job cannot be
  purged: its history is not disposable while its outcome is still open.
- Nothing expires on its own. Retention is an operator decision, and a job
  removed from the ledger is gone from the audit trail with it.

## Recovering by hand

The ledger is ordinary SQLite. To see what a job did:

```bash
sqlite3 "$JOB_ROOT/job_ledger.sqlite3" \
  "SELECT sequence, from_status, to_status, at_utc, actor, reason
     FROM job_transitions WHERE job_id = 'sj_…' ORDER BY sequence;"
```

Read freely. Do not write: the transition log is append-only by trigger, and a
status edited around the state machine is exactly the corruption the machine
exists to prevent. Resolve an `unknown` job through
`manager._ledger.transition(job_id, "interrupted", reason="verified by …")`
once you have established what happened to its supervisor.

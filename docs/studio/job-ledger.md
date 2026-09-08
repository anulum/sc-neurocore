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

Durability is a property of a **configured** job root
(`SC_NEUROCORE_STUDIO_JOB_ROOT`). Without one, Studio creates a private
directory for the process and its records go with it: an unconfigured root is
scratch, and `GET /api/studio/jobs/status` reports `configured: false`. One
fixed shared path would otherwise hand every Studio on the host the same
ledger — a second process reporting a first one's jobs as its own, and on a
multi-user machine a directory owned by whichever user created it first.

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
| Lease held by a process still running | Leaves the job alone; reports an expired lease if present |
| Lease held by a process on this host that is gone | `interrupted` |
| Lease held by a supervisor this host cannot probe, even if expired | `unknown` |

A lease stamped with this process's identity can belong to a job it is still
running. Recovery probes that identity normally: opening another manager over
the same root or calling `reconcile()` does not imply a restart. The identity
includes the process-start token, so a reused PID cannot inherit a dead
process's jobs.

Lease expiry alone is not evidence that computation stopped. A live process may
be busy without a heartbeat; recovery must not seal its outcome or permit its
artifacts to be purged while it can still write them. An unprobeable supervisor
stays `unknown` until there is evidence of its outcome, even after lease expiry.

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

## Bounded compute and cancellation

A Studio that accepts every submission and starts it immediately lets one
caller decide how much of the machine it uses. Admission bounds that: a fixed
number of jobs run at once, a fixed number wait behind them, and a submission
that arrives when both are full is refused with `job_queue_full` and the
counts that caused it. A refused submission never reaches the ledger — it did
not happen. `GET /api/studio/jobs/status` reports `admission` with the running,
queued, admitted and refused counts.

### Stopping a job stops what it started

A process job's worker leads its own process group. Cancelling or timing one
out signals the **group**: SIGTERM first, so a worker that handles it can seal
its own files, then SIGKILL after the grace period, then a check that nothing
in the group is still running. A worker that spawned children takes them with
it. The reap reports what it achieved and never raises, because a supervisor
that crashes while cleaning up leaves a job with no terminal record at all.

If a group survives even SIGKILL, the job's error says so and names how many
processes may still be running. That is a fact an operator can act on; silence
would not be.

### Thread jobs need a cooperative task

A Python thread cannot be killed. A thread task that never checks
`context.cancelled` keeps running after its deadline, and the ledger cannot
change that — so it does not pretend to. Such a job is recorded as `timed_out`
**and** its error states that the worker did not stop and that uncooperative
work belongs in a process job. The job id also appears in
`unreaped_workers` on the status payload.

Write thread tasks that check `context.cancelled` in their loop, or submit them
as process jobs.

### Nothing arrives after the outcome

The terminal transition is the seal. A worker that outlives its deadline cannot
post a result afterwards: the state machine refuses a second terminal
transition, so a late success is rejected rather than overwriting the timeout
that was already reported.

### What a request is allowed to cost

The synchronous budget projects a request's cost from the model it actually
names: the effective timestep resolved through the run contract, the
integrator's substeps and the declared state count, not `ceil(duration / dt)`
with a reference timestep. A four-state conductance model with a substepped
integrator costs hundreds of times a scalar map, and the budget now says so
instead of admitting both as if they were the same work. A request naming no
resolvable model keeps the scalar weight.

The catalogue scan honours cancellation between models. A cancelled sweep
raises rather than finishing all 185 and being discarded, and nothing partial
is cached: a partial sweep is not a scan of the catalogue.

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

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

Schema `studio.job-ledger.v7`. Single host by design: SQLite in WAL mode
serialises the writers sharing one root. A distributed worker contract is a
separate obligation and is not implied here.

Version 7 adds a nullable, validated training configuration snapshot to the
job row. Training start and weight-restore attach store the resolved
`studio.training-config.v1` payload in the same transaction as admission.
The stored JSON is canonical and capped at 4096 bytes; a malformed stored
snapshot makes reads fail closed. Earlier jobs retain a null snapshot rather
than a guessed configuration. `GET /api/training/jobs` reads the ledger when
Studio has a configured manager, so a restarted API can list retained jobs in
creation order. The historical local-thread facade still reads its in-memory
registry when called without a manager. This schema addition does not
authorize a separate-UID worker or cross-user access.
The complete job-record snapshot also carries the nullable field; the storage
record decoder refuses a malformed, oversized or wrong-kind configuration
instead of dropping it in transit.
The exact peer record wire contract is `studio.storage.record.v2`, and the
operator list payload is `studio.jobs.list.v2`; a v1 peer must fail closed
rather than accept a changed record shape under its old version.

Version 6 adds a storage admission replay table. A trusted storage authority
can bind a requester, workspace and mutation identifier to the exact request
digest and the original admitted or refused outcome in the same transaction as
job admission. This does not change the existing job idempotency key, and it
does not enable the isolated runtime by itself. The service must validate and
digest the complete versioned request, enforce policy, own worker custody and
qualify its socket boundary before accepting storage mutations.
Storage replay admissions reject a simultaneous legacy idempotency key: that
older lookup has no request digest and cannot prove an exact mutation replay.

Stored artifact manifests must be JSON arrays. A scalar, object or `null` is
corruption, not an empty manifest. Record reads, listings and reconciliation
raise `StudioJobLedgerCorrupt` without rewriting the malformed column or its
transition history. Preserve the database for diagnosis; do not replace corrupt
custody with an empty list to make recovery proceed.

The ledger transaction context rolls back an active transaction when either its
body or SQLite `COMMIT` fails, then propagates the error. A commit refusal must
not leave the connection holding a write lock or expose uncommitted changes as
durable state. This database rollback does not undo filesystem operations: a
purge whose cleanup commit fails retains its journal for the conservative
recovery rules below.

Execution timeouts in runtime settings, manager construction and thread/process
submission must be finite positive seconds. `NaN`, infinity, zero and negative
values are refused before admission or worker startup; invalid manager defaults
are refused before the job root or ledger is created. A submission override of
`None` uses the validated manager default, not an unlimited execution deadline.

`manager.wait(job_id, timeout_seconds=...)` observes the durable record even
when another process submitted it or this manager was restarted. Local events
can wake it sooner; otherwise it checks the ledger every 50 ms. A finite wait
deadline returns the latest record, which may still be running. It does not
cancel, reconcile or re-execute the job. Zero or negative timeout reads without
waiting; `None` waits for a terminal outcome. Non-finite timeouts are rejected.

Durability is a property of a **configured** job root
(`SC_NEUROCORE_STUDIO_JOB_ROOT`). Without one, Studio creates a private
directory for the process and its records go with it: an unconfigured root is
scratch, and `GET /api/studio/jobs/status` reports `configured: false`. One
fixed shared path would otherwise hand every Studio on the host the same
ledger — a second process reporting a first one's jobs as its own, and on a
multi-user machine a directory owned by whichever user created it first.

## What a job carries

Quarantine archive creation and restore use named process jobs. The API retains
its administrator policy, audit export and pre-submission restore validation;
workers receive JSON snapshots, not live audit-sink or ledger objects. The
existing writers produce the archive/restore bytes and SHA-256 manifests, and
the restore writer revalidates the archive/manifest pair in the worker. Job
service owners, request IDs, receipt schemas and artifact paths are preserved.
These routes still wait for their job outcome; process dispatch does not turn
them into asynchronous-receipt HTTP APIs. Embedded execution remains same-UID,
so this object boundary does not establish filesystem isolation.

`POST /api/models/scan/jobs` runs the complete catalogue scan in a named
process worker at the existing operating point: current `10.0` in each model's
native input units and duration `100.0` ms. Its receipt remains
`studio.model-scan.job.v1`; results remain `studio.model-scan.v1`, including
per-model errors and configuration/result digests. Successful job completion
does not imply that every model was classified successfully: inspect
`error_count` and `failed_models`.

Each scan job has its own process-local scan cache; separate asynchronous jobs
do not reuse the API interpreter's cached classifications. The synchronous
budgeted scan and its cache are unchanged. Cancellation of a process scan is
supervisor-enforced, including while a single model is being evaluated.

`POST /api/analysis/jobs` submits a named process task for simulation, F–I
curves, bifurcation sweeps, heatmaps and sensitivity analysis. The worker
revalidates the request and executes the same analysis implementation as the
synchronous surface. Parameter order travels as an explicit sequence, so JSON
key sorting cannot change the ordering of equal sensitivity values or the
result's evidence seal. The public receipt remains `studio.analysis.job.v1`;
job records identify `execution_model: process`.

Process execution is not an operating-system isolation guarantee. Embedded
workers still use the launching user's identity; the isolated storage mode and
its limits are described under [Storage mode and isolation](#storage-mode-and-isolation).

An analysis cancellation recorded by another manager sharing the ledger is
observed by its owning supervisor. Cancellation and execution timeout stop the
registered worker's process group; they do not publish a successful analysis
result. Capacity becomes reusable after terminal-state and admission cleanup.
Neither a completed status read nor an observation timeout should be used as a
substitute for worker termination evidence when performing recovery or cleanup.

If the supervisor dies, recovery records the unfinished analysis as
`interrupted`; it does not replay the request or promote a worker result to
success. The worker lifetime guard stops the orphan independently of a new
manager. Until the worker group is proved stopped, its admission slot remains
held—even if the job already has an `interrupted` record. A stopped (`SIGSTOP`)
process is still alive and does not qualify for capacity reclamation.

| Field | Meaning |
|---|---|
| `owner`, `workspace` | Who ran it and in which scope. Reads can be scoped by both |
| `request_id` | The caller's correlation id; HTTP analysis and catalogue scan jobs retain the middleware-normalized request trace |
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

The Studio analysis and catalogue-scan controls show these recovery states
without treating them as successful results. They keep polling an `unknown`
record and block another submission from that control while its outcome remains
unverified. An `interrupted` record stops the poll and retains no result.

A job asked to cancel before its supervisor marks it running stays
`cancelling`: it did start, and it is already winding down. A terminal record
is never rewritten, so a late claim of success on an interrupted job is
refused rather than accepted.

Repeating the same terminal status is an idempotent no-op only when every
supplied result, artifact manifest, error and start/finish timestamp matches
the sealed record. Omitting a field leaves it unchanged. A conflicting retry
is refused inside the ledger transaction; neither the record, heartbeat nor
transition history changes. Recovery cannot silently replace a sealed result.

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

Recovery rechecks its observed record inside the transition transaction. If a
result, status, lease or heartbeat changed meanwhile, it retains the newer
record instead of applying the stale decision. A concurrently purged job is
omitted and never recreated. Recovery reports the retained status, including
completion committed by the actual supervisor; it does not invent that result.

Only the recorded supervisor may renew a live lease. A heartbeat from another
supervisor is rejected even when the lease has expired. Control or recovery
transitions from another supervisor preserve the original heartbeat and lease
expiry rather than claiming fresh evidence of worker liveness. Terminal
transitions still clear the lease; this is not a lease-transfer mechanism.

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

Managed process workers arm an independent interpreter in their dedicated process
group after ledger registration and before importing task code. Startup allows
at most three seconds for its readiness token. Invalid, missing or incomplete readiness refuses
task execution and kills/collects the guard child. The outer supervisor retains
whole-group cleanup responsibility if that direct-child cleanup fails. The guard observes the
recorded supervisor every 100 ms and kills its own group when that supervisor
dies, including when the supervisor is a zombie awaiting collection. Unknown
liveness is tolerated for one second before stopping the group. This guard does
not depend on the task releasing the Python GIL. Normal supervisor reaping also
removes the guard; standalone workers without managed supervisor registration
do not acquire one.

These are polling intervals, not hard real-time guarantees. A stopped or
unscheduled guard cannot enforce them, and processes that deliberately escape
the session are outside this containment contract. Recovery must still verify
the recorded group's termination before releasing capacity; guard startup or
an `interrupted` record alone is not proof of cleanup. Unavailable process
metadata yields unknown liveness rather than an invented live owner.

Stop requests are durable: both thread and process supervisors observe the
shared ledger, so a second manager can cancel an existing job. A repeated Stop
also delivers the owning manager's local cancellation event. The request is
recorded before that event is delivered, so Stop on a live job answers
`cancelling` and its history shows the request before `cancelled`; a job that
stops on its own before the request is recorded answers with the state it
reached. The event is delivered even when the ledger write fails. Cancellation is
cooperative for threads; it is a request, not proof of completion.

If cancellation-state observation fails, the supervisor first requests thread
stop or reaps the process group, then attempts to record `failed` with the
cleanup outcome. An uncooperative thread is explicitly reported as still alive.
If the ledger also refuses that final write, no terminal outcome is claimed;
the durable record needs recovery after storage is repaired.

A Studio that accepts every submission and starts it immediately lets one
caller decide how much of the machine it uses. Admission bounds that: a fixed
number of jobs run at once, a fixed number wait behind them, and a submission
that arrives when both are full is refused with `job_queue_full` and the
counts that caused it. A refused submission never reaches the ledger — it did
not happen. `GET /api/studio/jobs/status` reports `admission` with the running,
queued, admitted and refused counts.

Capacity belongs to the **shared job root**, not to each manager or API process.
The first submission persists `max_concurrent_jobs` and `max_queued_jobs`.
Opening an observer neither establishes nor changes these limits. A new
submission configured with different limits is rejected; an idempotent lookup
of an existing job can still return it without admitting work.

Reservation, job creation, initial transition and admitted-counter increment
commit together. If creation fails, all four roll back. A repeated key is scoped
by actor and workspace, returns the existing job and consumes no additional
capacity. Queued callers recheck that key before promotion, so two waiting
copies do not execute twice. Distinct queued requests advance by reservation
ticket order. Submission waits synchronously for admission; the job execution
timeout starts after admission and does not bound time spent in this queue.

Running counts include capacity retained for unreaped workers. A terminal
record alone is not permission to release its slot. Reservation release is
supervisor- and job-scoped; another manager cannot free that slot. Counters
persist across manager restarts; migrated historical jobs are not retroactively
counted as newly admitted submissions.

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

Normal worker exit also checks and stops remaining members of its group before
recording completion. If cleanup fails, the outcome is `failed` and capacity
stays occupied, even if the direct worker produced a successful result. This
contract covers the worker's process group, not descendants that deliberately
escape into another session; it is not an OS sandbox or distributed executor.

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

Version 2 adds `admission_config` and `admission_reservations` inside the same
ledger. The migration and version update are transactional: a failed migration
leaves version 1 retryable, without partially installed capacity tables. Existing
job rows and transition evidence are retained. Pending, running, cancelling and
unknown jobs occupy capacity; legacy errors explicitly reporting a worker that
did not stop or a group that was not reaped retain an unreaped reservation.
These historical error markers are not a complete inventory of orphan processes.

For an upgrade, stop accepting submissions and quiesce all old managers sharing
the root. Preserve a consistent ledger **and its job directories** before opening
it with the new build. Verify the stored schema and retained records on that
backup first. Do not mix old and new manager implementations over a migrated
root: old in-memory admission does not enforce the new root-wide limits.

There is no in-place downgrade. To roll back, stop new managers and preserve
the entire post-upgrade root, then restore the verified pre-upgrade root with
its matching older build. Jobs or evidence produced after that backup must be
retained separately and reconciled; they are not present in the restored state.
Do not reset `schema_meta` or delete reservation tables to bypass version checks.

After job reconciliation, `manager.reconcile()` reclaims a queued reservation
whose supervisor is proven dead and which never created a job. It also reclaims
terminal thread work whose supervisor is proven dead, because its in-process
threads cannot survive that death. Unknown identities and expired leases alone
do not justify release. Process reservations remain held after supervisor death:
the worker or its descendants may still be alive. Historical workers lack the
identity evidence needed for automatic orphan cleanup after restart.
Do not delete those reservations or equate `interrupted` with a stopped worker.

Version 3 adds `job_workers`. The trusted parent registers its observed child's
supervisor, host/PID/start token, boot identity and session group before permitting
task import. It captures the PID generation before its monitoring loop can reap
the child. The managed worker receives a fixed grant over its private stdin pipe;
it does not open a ledger based on its work directory. EOF, malformed input or a
three-second wait without the grant refuses execution before task import. Managed
stdin is closed after this handshake and is not an interactive task input stream.

Registration runs in a separate parent thread so a SQLite writer-lock wait does
not stop timeout/cancellation monitoring. Late registration rechecks captured
identity and current admission; a dead or inactive worker receives no grant.
Startup failure closes the pipe and reaps the already spawned group. The existing
independent lifetime guard is armed after registration and before task import.
This is a trusted-parent local bootstrap, not a different-UID sandbox or an
authentication mechanism against arbitrary same-UID processes. Standalone workers
without managed supervision retain their separate unmanaged execution mode.

Registration refuses absent capacity or inactive/mismatched supervision. Existing
version 1/2 jobs receive no invented worker identity. Registration records startup
custody, not completion. Once the supervisor is proven dead and the job is terminal,
reconciliation can release its reservation when matching worker evidence proves
a previous local boot or a stopped local group. Live groups, including reused
group IDs, remain held. Missing, foreign, malformed or unreadable identity evidence
does not authorize release. This reconciliation never kills an orphan and does
not change the sealed job result.

## Retention and backup

### Connected storage read building block

The settings builder accepts optional `SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY`
JSON only alongside `SC_NEUROCORE_STUDIO_STORAGE_MODE=isolated`. Every boundary
field is explicit: `storage_uid`, `api_uid`, `worker_uid`, `authority_root`,
`spool_root`, `socket_path`, `workspace`, `frame_max_bytes`,
`max_metadata_bytes`, `max_seed_bytes`, `max_seed_entries`,
`max_manifest_bytes`, `max_artifact_bytes`, `max_artifact_entries`,
`transfer_timeout_seconds` and `max_connections`. The artefact budgets bound
the bytes and entries a finished job may seal into the storage authority.
The metadata limit cannot exceed one frame, and the manifest-name budget
cannot exceed the metadata limit. UID values must be distinct
non-root OS identities. Storage, spool and socket-parent trees must be absolute,
canonical and disjoint. Invalid fields, duplicate JSON keys and invalid limits
refuse configuration; no paths are created or permissions repaired.

These settings alone do not make a working deployment: an isolated API starts
only after the preflight under [Storage mode and isolation](#storage-mode-and-isolation)
passes. Path validation does not prove ACL, mount or inherited-descriptor
isolation. Do not enable an embedded fallback or treat this configuration as a
production migration procedure.

`storage_namespace.open_storage_directories` is a read-only startup prerequisite
for existing authority and socket-parent directories. It verifies the configured
service identity and walks each ancestor using non-following, directory-relative
opens. Authority ownership must match the service with mode `0700`; endpoint
parents cannot be writable by group or others. Ancestors must have trusted
root/service ownership, with sticky protection where otherwise writable.
Its borrowed, noninheritable descriptors close on acquisition failure or context
exit. It does not create paths, repair permissions, open a ledger or bind a
socket. These checks do not replace mount/ACL/capability policy or actual
different-UID isolation qualification.

`storage_record_client.read_storage_record` exchanges one bounded record request
with `storage_record.serve_record_read` on an already connected Unix stream.
Both endpoints verify the configured OS peer before transferring frames. The
authority applies the existing operator policy before reading the ledger; the
client validates the response version, trace, complete job snapshot, job ID and
workspace. Denied or missing reads never return a partial record. Both endpoints
close their single-use socket on success or failure; no automatic retry or local
ledger fallback occurs. A shared absolute monotonic transfer deadline bounds
the request and response, not SQLite or audit-sink execution.

The storage service's listener dispatches record, admission, supervision,
finish, query, cancel, artefact and purge operations. The API forwards a
request's principal only after that request's route policy allowed it, and the
service applies its own policy again. Same-UID socket tests do not prove
protection from other processes with those filesystem permissions.

The named-admission client can send one strict versioned request and exact seed
bytes over an already connected, peer-verified Unix stream. It checks the
configured workspace, metadata size, canonical seed manifest and actual byte
lengths before sending the first frame. The service independently checks the
API peer and process generation, the reviewed task and route policy, then
derives replay identity from received bytes, then admits the job through the
shared admission ledger and answers with the correlated admission result below.

The versioned admission-result codec accepts a job ID or queue-full refusal
from the authority's transaction; the service must establish worker custody
and durable outcome before sending it. The API-side reader verifies the service
peer and binds the reply to the trace, mutation ID, workspace, reviewed task and
content digest retained from the exact metadata bytes sent. Sending and
receiving share one absolute deadline; the reader closes the stream on every
outcome. A lost reply remains ambiguous, so a caller can deliberately replay
the same mutation ID and content against the durable authority. The API must
read the full job record separately after an admitted job ID. In isolated mode
the API's job facade submits every job through this path.

Version 4 introduced purge intent in `job_purges` before moving any artifact
directory. The journal records the purger, directory device/inode and whether
database deletion committed. On startup, recovery of a proven-dead purger
restores staged artifacts if the job remains, or finishes staged cleanup after
committed deletion. It uses the journal and directory identity, not filename
matching alone. Conflicting paths or unproven ownership remain pending rather
than being overwritten or deleted. Filesystem and database operations are not
one atomic transaction; the journal supports recovery but cannot prove every
interrupted cleanup completed.

Version 5 extends that same journal with durable cleanup phases. Migration keeps
all existing rows and identities unchanged; it never assigns removal evidence to
an old row. Migration failure rolls back the schema change. Custom journal
indexes or triggers require review and cause automatic migration to refuse;
they are not silently dropped. Older writers must not open the newer schema.

| Phase | Durable meaning and recovery |
| --- | --- |
| `prepared` | Directory custody recorded; restore retained job when safe. |
| `committed` | Job deletion committed; cleanup has not yet been recorded as started. |
| `cleanup_started` | Committed before destructive cleanup; resume only against exact recorded custody. |
| `removed` | Verified object removal and parent directory sync observed before committing this phase; journal closure can retry separately. |
| `ambiguous` | Missing or conflicting custody without sufficient proof; never resolve automatically. |

Each phase rechecks the current row and supervisor ownership under a SQLite
writer transaction. Another recovery pass cannot act using a stale phase.
Cleanup-start, removal evidence and journal closure use separate commits, so a
failed final journal commit can retry from durable `removed` evidence. The
filesystem/SQLite gap before that evidence commits remains real.

Staging and restoration require Linux `renameat2(RENAME_NOREPLACE)` support
from libc, kernel and filesystem. A destination appearing after inspection,
including an empty directory, is not overwritten. Unsupported operations or
other OS errors fail without a non-atomic rename fallback. If nothing moved,
recovery can cancel the prepared intent while preserving the job and its files;
otherwise the journal remains until restoration or cleanup is verified.
Directory entry changes are synchronized before committing database deletion
and before removing a resolved journal entry. Sync errors propagate instead of
claiming durable completion. These ordering guarantees still depend on the
filesystem and storage device honoring synchronization requests.
After a move, the destination identity is checked again before deleting a job
record or completing restoration. A substituted source leaves recovery pending;
the move alone does not prove that the authorised directory was recovered.
Committed cleanup verifies an opened directory's device/inode and removes its
contents relative to that descriptor, without following root symlinks. After
the directory is opened, replacing its pathname does not redirect those
descriptor-relative content operations. Device/inode matching does not prove
identity across deletion and recreation: the filesystem can reuse an inode
before cleanup opens the directory, and a replacement can then pass this check.
After removing the root pathname, cleanup checks that the verified open
directory has zero links. Successful removal of a replacement is not reported
as successful deletion of the original object. This detects false completion;
it does not prevent an untrusted process with namespace write access from
replacing that pathname or exploiting inode reuse before the open. The embedded
mode requires all writers to cooperate with the ledger protocol; it does not
isolate storage from other processes with the same filesystem permissions.
Do not edit, move or recreate job and staging directories outside that protocol
while the service is running. Storage-owner isolation is required if writers
cannot be trusted to respect this boundary.

If the journal recorded a directory identity but its staging path is absent
without a committed `removed` phase, recovery records `ambiguous` and retains it
across reconciliation and restart. It cannot distinguish successful deletion
from a moved-away original. This includes a crash before removal evidence commits
or a post-removal directory sync failure. A directory already absent when purge was prepared has no recorded
identity and follows the separate no-directory cleanup path. Do not manually
delete an ambiguous journal row to clear the counter. Preserve the ledger and
remaining filesystem evidence for operator investigation; automatic resolution
of missing-stage ambiguity is not implemented.

`GET /api/studio/jobs/status` includes `pending_purge_count`, the number of
durable purge intents awaiting completion or restoration. This additive field
does not expose paths, process identities or individual job IDs. Reading status
does not perform recovery. A nonzero count warrants inspection before declaring
storage cleanup complete; older servers may omit the field.

For operator inspection, `GET /api/studio/jobs/purges?limit=100&after=<job_id>`
returns `studio.jobs.purges.v1`: a `purges` list containing `job_id`, `phase`,
recorded `device` and `inode`, and `next_after` (null on the last page). Omit
`after` for the first page. The default limit is 100; accepted limits are 1–1000.
Job-ID ordering is lexical, not chronological. Pages are live database reads,
not a frozen multi-request snapshot. Invalid limits or cursors return 422.

The route uses the existing ADMIN policy and `studio.jobs.purges.list` audit
action. Deployments must enable route-policy enforcement; policy-disabled local
mode is not an authenticated boundary. An audit failure refuses the read. This
global operator view exposes no storage paths or supervisor/process identities.
It performs no reconciliation, filesystem inspection or journal mutation. A
recorded inode is evidence to investigate, not permission to delete that object.
Python operators can use `manager.purge_snapshot(limit=100, after=None)` for the
same bounded read; the local facade itself does not authenticate callers.

After correcting a transient filesystem error, call `manager.reconcile()` on
the same supervisor process to retry its committed purge cleanup. This works even when
the job record has already been deleted. A successful retry removes the intent;
repeated cleanup errors propagate and leave it pending. The return value lists
job-recovery decisions, not purges: read `pending_purge_count` again afterwards.
Reconciliation does not take over another live supervisor's purge, or restore its
own prepared operation while that operation may still be executing. A new
supervisor process can recover the earlier process's intents only once its
supervisor is proven dead. Managers in one process share a supervisor identity.
Reading HTTP status does not trigger these operations.

- The ledger is one file. Back it up with the job root it describes; separating
  them turns records into orphans and artifacts into anonymous bytes.
- WAL mode leaves `job_ledger.sqlite3-wal` and `-shm` beside it. Copy the
  ledger only while no Studio process is writing, or use `sqlite3 … ".backup"`,
  which is consistent under concurrent writers.
- `manager.purge_terminal_record(job_id)` deletes a terminal job's directory,
  its record, worker identity and whole transition history. An unfinished job
  or a job retaining a capacity reservation cannot be purged. This check applies
  before directory deletion and inside direct ledger deletion; terminal status
  alone does not establish that a worker stopped. Reconcile proven-stopped work
  before retrying a refused purge.
- Nothing expires on its own. Retention is an operator decision, and a job
  removed from the ledger is gone from the audit trail with it.

## Storage mode and isolation

`SC_NEUROCORE_STUDIO_STORAGE_MODE` selects the storage trust model independently
of the HTTP deployment profile. An absent value or `embedded` keeps the existing
local ledger and job-root behavior. Embedded mode is **not** an OS isolation
boundary, including when the deployment profile is `production`.

`isolated` separates three Linux identities. The storage service alone owns
the ledger and sealed artefacts; the API holds no job root or ledger; a
privileged launcher starts each worker under a compute identity. The API reads
`SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY` (above) and
`SC_NEUROCORE_STUDIO_STORAGE_LAUNCHER`, strict JSON with `socket_path`,
`launcher_uid`, `worker_gid`, `grant_timeout_seconds`, `heartbeat_seconds`,
`poll_seconds`, `attempts`, `live_retain` and `accept_direct_backend_limits`.
The operator starts `python -m sc_neurocore.studio.platform.storage_service
--configuration PATH` (the boundary plus `max_concurrent`, `max_queued`,
`audit_log_path` and `reconcile_seconds`) and `python -m
sc_neurocore.studio.platform.storage_launcher_service --configuration PATH`.
Unknown or empty mode values are configuration errors.

Before constructing any collaborator, an isolated API checks that a boundary
and launcher are configured, route policies are enforced, header identity is
off, an identity store and a persistent audit log are configured,
`fs.protected_hardlinks` and `fs.protected_symlinks` are `1`, the process runs
as the configured API identity, the spool root is API-owned, closed to others
and traversable by the compute group, the API belongs to that group, the
launcher endpoint directory exists, and the direct backend limits below were
accepted. Startup refuses with every failed check named. Nothing is repaired
and there is no fallback to embedded storage.

The only launcher backend spawns each worker directly. It proves neither that
every forked or detached descendant stops with its job nor per-job resource
accounting: its process limits apply per compute identity, not per job. The
API therefore refuses to start unless `accept_direct_backend_limits` is
explicitly `true`, which suits an evaluation or qualification deployment. An
isolation claim that needs guaranteed descendant termination or per-job
accounting requires a cgroup-based backend, which is not yet available.
Setting this option does not migrate existing data, change permissions, create
host accounts or qualify an isolated deployment.

## Recovering by hand

Evidence bundle exports now use a named process task. The API supplies complete
public job snapshots and copies verified source artifacts into deterministic
submission seeds one at a time. The worker cannot use this envelope to choose
a ledger path or reader callback; it rechecks snapshot types, aggregate bytes
and source digests through the original bundle writer. All existing evidence
categories remain supported.

`SC_NEUROCORE_STUDIO_EVIDENCE_MAX_INPUT_BYTES` sets the aggregate export input
ceiling in bytes (default 268435456, or 256 MiB). The count includes encoded task
metadata and every declared binary seed copy; repeated selections also count.
The per-artifact limit still applies independently. An oversized selection returns
HTTP 413 with `studio_evidence_input_limit_exceeded` before job admission, without
dropping inputs. Invalid explicit limits refuse startup. This is a configurable
operational budget, not a scientific restriction or a total process-RAM guarantee.
Source transfer/integrity failure returns `studio_job_failed`; a failure after
admission retains its failed job and may retain partial seed files. No automatic
source deletion or migration occurs. Same-user processes remain non-isolated.

Seed reads and existing-artifact publication read at most the configured size
limit plus one byte before rejecting oversize data. Declared artifact reads use
the retained declaration's size plus one byte, then verify its size and SHA-256.
This prevents an enlarged file from forcing an unbounded read; it is not an
aggregate job-memory or cross-user filesystem isolation guarantee. A lowered
write ceiling does not invalidate an older, correctly sealed artifact.

Evidence bundle exports bind each copied artifact to the captured source job
record. A reader returning different path, size or digest metadata is rejected;
the returned bytes are also independently checked against that declaration.
A mismatch fails the export without a final manifest, although earlier partial
entries can remain in the failed job. This is not atomic bundle publication.

Training weight restoration runs as a named process task. Metadata and checkpoint
bytes enter through bounded submission seeds; the original materializer verifies
their sizes and digests before the restricted torch loader executes. The worker
returns the existing restore evidence, not tensor state. This preserves the
synchronous HTTP response contract and does not establish OS-level isolation.

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

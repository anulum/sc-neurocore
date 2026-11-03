<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Native Go MAT statement gate -->

# Native Go MAT statement gate

`tools/go_mat_coverage.py` checks the complete maintained Go MAT family using
Go's real atomic counters. It covers both simulation services and both raw C
exports. The original statement floors are 50, 45, 26 and 23 respectively.
Every statement-bearing block must execute. Both profiles retain their full
native block universe, including zero-statement blocks and unrelated native
packages; the comparison refuses changed coordinates or denominators.

These are Go **statement** measurements. They do not establish general branch
coverage, MC/DC, scientific model fidelity, isolated performance, GPU readiness
or whole-project qualification. The two empty `main` functions have no
statements and remain visible in the full native function reports.

## Run inside an admitted allocation

Use an unoptimized Linux Python with its installed GNU Readline module, this
checkout's normal development dependencies and the exact Go version declared in
`src/sc_neurocore/accel/go/go.mod`. Build the
two Mojo MAT images from the same checkout in the same job, using the existing
pinned Mojo accelerator producer. CI already builds these images before the
native gate. An image downloaded from another job or retained from a different
source revision does not satisfy that producer requirement.

Resolve the following arguments from the executing host's admitted profile:

```bash
python - "$CHECKOUT" "$ALLOCATION/python-coverage" <<'PY_TRACE'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from tools.go_coverage_trace import write_config
write_config(Path(sys.argv[1]), Path(sys.argv[2]))
PY_TRACE
python -I -B -m coverage run \
  --rcfile "$ALLOCATION/python-coverage/go-mat.coveragerc" \
  "$CHECKOUT/tools/go_mat_coverage.py" \
  --repository "$CHECKOUT" \
  --output "$ALLOCATION/go-mat" \
  --go "$ADMITTED_GO" \
  --seconds "$ADMITTED_SECONDS" \
  --jobs "$ADMITTED_COMPILE_JOBS" \
  --stop-seconds "$ADMITTED_STOP_SECONDS" \
  --python-coverage-config "$ALLOCATION/python-coverage/go-mat.coveragerc"
```

The output directory must be new, outside the checkout, on admitted storage,
and contain no comma (Go's native input-list separator). The tool creates
exclusive command receipts and a separate native-artifact namespace. An
existing output allocation is refused. It does not grant resources, acquire a
machine lease, change cooling or qualify a recovery controller. The caller
retains responsibility for the complete allocation, memory/disk limits and
final cgroup closure. Where the executing host provides a cooling or thermal
gate, apply it before starting native compute.

The hosted CI adapter derives compile concurrency from actual process affinity
and every visible ancestor CPU quota. It records missing controller files
explicitly; Linux documents `cpu.max` as a non-root cgroup interface. See the
[upstream cgroup-v2 specification](https://raw.githubusercontent.com/torvalds/linux/master/Documentation/admin-guide/cgroup-v2.rst).
The runtime policy assigns one quarter of the enclosing job's remaining
recorded budget to this gate and one two-hundredth of that gate budget to stop
grace. These fractions are explicit policy, separate from host capacity. The
recorded clock starts in the first owned test-job step before checkout and
toolchain setup; it is not GitHub's service-side job-start timestamp. The
profile also retains actual memory and disk headroom, without claiming a
memory/disk reservation or resource admission.

The live collector always reads its actual Linux cgroup, affinity, memory and
disk interfaces. Separate typed validators accept those raw observations;
refusal contracts damage owned copies of kernel metadata and never replace
an OS provider or claim that a copied hierarchy is a real capacity profile.

## What executes

The gate builds ordinary and coverage-tagged Go images. Actual `nm -D`
inspection requires the production simulation exports in both and the
diagnostic snapshot export only in tagged images. Actual `go list -deps`
checks that `runtime/coverage` and the internal snapshot adapter are absent
from ordinary builds. Go formatting and normal/tagged vet checks are required.
The diagnostic Go files all carry `//go:build coverage`.
An additional real tag-only build omits `-cover`; its present export must still
refuse a snapshot. Thus the tag and actual native counter instrumentation are
independently required.

The actual before-work process loads the new Go images and collects native
snapshots without calling the model. Its public statement verdict must return
OS status 1 with zero covered subject statements. The complete after-work
profile combines all thirteen family Go tests (nine original and four public
simulation-helper regressions) with the unchanged complete 630-case Go/Mojo
C ABI module. No test is skipped, deselected or replaced.

The four helper regressions compare full Python-generated MAT and resetting
MAT voltage traces and exact event totals, and preserve both empty-helper
results. They call the actual Python classes from this source checkout; no
copied recurrence provides the oracle. Their workload and numerical tolerance
retain the existing qualification protocol.

The ABI consumer collects all 630 unique nodes before execution and requires
the same full membership at execution. It retains JUnit and actual pytest
reports. All 72 original crash-fenced null-pointer launches must match distinct
registered children by actual argv and parent PID, with their declared image
roles actually loaded. Their unchanged tests assert each actual child status
is zero; the observer does not invent a separate child exit-code record.

## Own Python statement and branch evidence

The existing hash-locked Coverage.py version traces the gate's nine tools and
complete ABI consumer: fourteen actual source files. Its explicit configuration removes line and partial
branch exclusion lists, and enables only the standard `_exit` patch. The
reviewed closed environment declares `COVERAGE_RUN=true` and the admitted
`GOMAXPROCS`; it does not inherit subprocess tracing configuration.

The real image observer is registered as a callable with `__cantrace__` enabled,
so standard Python tracing records the actual audit callback and child-launch
validation. [CPython's audit-hook tracing protocol](https://docs.python.org/3.12/library/sys.html#sys.addaudithook)
otherwise omits hooks from tracing even while they execute. This changes no
image, environment or refusal check and supplies no manufactured counters.

CI joins the Linux gate's own counters with real unsupported-platform API/CLI
refusals from the independent `native-go-windows-refusal` job. It has its own
checkout, pinned Python/Go, hashlocked dependencies, counter directory and
artifact, outside SHD coverage. Its labelled 20-minute governance ceiling
reserves four minutes for clock-origin skew, cancellation and evidence upload;
the inner timeout also subtracts actual elapsed setup time. Failed or timed-out
refusals retain their own budget and terminal records. The aggregate requires
this job as well as the original SHD leg and Linux shards to succeed.
Source hashes and the locked coverage version must match before recorded
workspace roots can become path aliases.
The Windows result is an unsupported-platform refusal, with no Windows native
backend qualification. A real provisional JSON report supplies report
conservation tests; an actual nonzero provisional verdict is retained. Counter
combination then appends those tests' observations and requires the complete
source cohort, no excluded statements, and the native coverage tool's final
100 percent statement/branch gate. No successful report is manufactured.

## Image, process and environment evidence

An explicit temporary bootstrap is generated only under the gate allocation.
It is never installed as a repository or global `sitecustomize.py`. The gate
passes a closed base environment to its own consumers. Before declaring it,
the producer initializes the actual installed Readline module used by Pytest
capture and reads its real `COLUMNS` and `LINES` from libc's environment. These
must be canonical positive ASCII integers; missing or invalid metadata refuses
the gate. Their actual values are declared in every consumer base, without an
invented terminal size. An exclusive first-initialization record retains the
actual module image/hash, producer PID, prior import state, Python-cache match
flags and declaration call stack. Every later declaration requires that same
module and terminal pair; later drift refuses before a consumer is launched. Readline can update the native environment while
Python's cached `os.environ` still lacks those fields. The declaration makes
the original inherited child mapping agree with both actual views. This is
terminal metadata, not a measured compute resource or admission limit.
Consumers start through a fresh exec with this explicit full mapping, so their
initial `/proc` environment and Python cache both contain the declared values.
Registration also rereads Readline's current C values and refuses any drift,
including a later native change that leaves Python's cache unchanged. Each
closed-environment record retains the actual terminal pair; independent final
reconciliation requires it to equal the complete declaration. Actual libc
missing/invalid-value and later-registration drift controls exercise refusal.
The original failed full ABI attempt and subsequent startup diagnostic remain
failed evidence; this correction supplies no retroactive qualification.
The unchanged original crash-fenced children inherit that base and Pytest's actual `PYTEST_VERSION`
and `PYTEST_CURRENT_TEST` values. Before each spawn the real parent audit
record declares the entire mapping and its digest. The version must match
the installed Pytest, and the marker must identify a crash-fenced call in the
actual complete 630-case collection. The child resolves exactly one declaration
by its actual PPID and argv. Its complete `/proc/self/environ` and Python
keys/values must equal that declaration; no fields are removed or ignored.
Final reconciliation binds all 72 actual child streams to their declarations.
Inherited `LD_*`, `PYTHON*`, `GODEBUG`, vendor credentials and plugins are not
accepted. The declared job bootstrap path is the sole explicit Python-path
exception. Actual interpreter executable bytes and prefix are checked.

Each MAT load records a full backing-file hash and stable device/inode
identity. Exit collection requires the matching executable mapping and
unchanged backing bytes. Each loaded Go image must actually refuse an invalid
snapshot destination, then produce both metadata and counters twice. The
second cumulative capture includes the first writer's completed success
return. These cumulative counters are never interpreted as unique API calls.
Every loaded Mojo image retains its executable-mapping observation.

## Productive refusals and retained output

The gate exercises actual missing-image, wrong-digest, ordinary-image,
closed-environment, missing-tool, corrupt-native-metadata, omitted-source and
zero-counter refusals. It also refuses a native output path that would collide
with command receipts **before launching a child**; that case has no invented
native exit status. Existing source, images and prior artifacts are preserved.

`result.json` is created only after the complete gate succeeds. Raw native
metadata/counters, complete text/function/debug reports, command argv/status,
process observations, source hashes and failed-control receipts stay under the
allocation. A failed job can retain partial artifacts without a success
result. Do not reuse its output directory or relabel an earlier failure.

The compressed reports under `tests/fixtures/go_mat_coverage/` are historical,
unfiltered native inputs for parser regression tests. Parsing them is not a new
native run and does not establish current coverage. Packaging, source quality,
own-tool statement/branch checks and hosted CI evidence are separate required
integration checks; the native statement result alone does not replace them.

Build observation validators consume the actual pinned formatter, symbol, dependency
and complete Go test outputs. Dedicated negative contracts damage copies of those
real observations; they do not replace tools, create successful test events or
replay numerical work to manufacture failures. A real owned child that ignores
SIGTERM checks the existing admitted stop deadline and actual SIGKILL/reap path.

Image validation keeps real descriptor reads, complete hashes, `/proc/self/maps`
and native writer calls in the production path. Negative contracts change only
unmapped owned file copies or copies of actual map, status and counter observations.
Loaded native code and original raw counters remain intact.

Control-result validation requires the exact actual refusal status, observation
stream and cause. Contrary-phase receipts and damaged copies must refuse. Producer
validation independently conserves original compiler declarations, zero before-work
subjects, actual transfer hashes and final source/bootstrap/image custody.

The complete original ABI test source remains byte-identical. Its consumer
records real collection origin and delegates unchanged admission checks to typed
validators. Negative record tests damage copies of the actual full630 result;
no Pytest Session or TestReport is fabricated. Separate small real Pytest cases
produce setup-failure, call-failure and skip reports for the observer control.
Those adverse cases claim zero original ABI executions and run no model workloads.
Pytest import remains after the actual closed-environment registration.

The native Go gate runs before the existing final Stage/Preserve shard-coverage
pair, whose original `if: matrix.coverage` conditions remain intact. GitHub
Actions applies its default `success()` status check to those conditions: if
the native gate fails, shard 8 fails and those two ordinary coverage steps are
skipped. Its ordinary shard-8 coverage artifact is therefore absent, and the
coverage union fails on the missing shard; partial ordinary coverage is not
reported as a successful union. The separate native-evidence upload retains
its explicit `always()` guard and keeps the failed native run's raw artifacts.

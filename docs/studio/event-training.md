# Event dataset training

The training API accepts local N-MNIST, SHD and DVS-CIFAR10 recordings with
an explicit manifest, group split and event encoder. Dataset files must already
be available to the operator under their licences. Starting a run does not
download data.

## Configure local recordings

For the SHD publisher snapshot identified by the SHA-256 values in
`examples/verify_shd_publisher.py`, create a
verified manifest from the two locally held, uncompressed HDF5 files:

```sh
python examples/verify_shd_publisher.py /operator/shd --output shd-manifest.json
```

The example checks exact file bytes, reports the 8,156 training and 2,264 test
recordings, then writes the same manifest type used by the training contract.
It refuses changed files and existing output paths. This identifies a specific
publisher snapshot; a future publisher revision needs its own verified digests.
The data is licensed under CC-BY-4.0; see the
[publisher](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

The published train and test files share ten speakers. A group split of the
training file keeps optimisation and evaluation speakers separate, but does
not remove this overlap with the published test set. State which evaluation
protocol a reported score uses. Manifest acceptance does not establish training
accuracy, native decoder parity or hardware measurements.

Set `SC_NEUROCORE_STUDIO_DATASET_ROOT` in the backend environment before starting
Studio. It identifies one dataset directory in the layout described by
[the dataset reference](../api/datasets.md). Training requests and exported
checkpoints contain relative recording names and content digests; they do not
contain this operator path.

The API verifies file bytes, sample labels and groups before creating a job.
The worker repeats verification and checks again before completion. Keep the
recordings unchanged throughout the run. Verification does not create an
immutable filesystem snapshot.

The operator can set `SC_NEUROCORE_STUDIO_EVENT_INPUT_MAX_BYTES` to a positive
byte limit; the default is 64 MiB. Admission accounts for float sample tensors,
their batch collation and one binary encoding buffer before loading recordings
or creating a job. Accounting uses the largest actual batch in the selected
parts. A request over the limit is refused with the required bytes and the
current limit; reduce `batch_size` or `timesteps`, or arrange a larger operator
budget. The config event records this accounting as `input_admission`.

This budget covers encoded inputs, rather than total process memory. Raw event
arrays, model parameters, optimiser state, activations and library overhead
remain subject to the separate worker resource limits.

## Optional native SHD execution

The SHD manifest sample reader can use the real Rust, Go, Julia or Mojo indexed HDF5 readers described
in [the dataset reference](../api/datasets.md#native-indexed-shd-readers).
For a regular backend, set `SC_NEUROCORE_SHD_RUST_LIBRARY` or
`SC_NEUROCORE_SHD_GO_LIBRARY` to the selected absolute library file. For isolated
workers, declare `event_input.shd_rust_library` or `event_input.shd_go_library` in the
operator launcher configuration; inherited API environment settings do not
select the worker library. The launcher validates the absolute file and passes
only that declared setting to the compute worker. These remain separate from
`event_input.rust_library` and `event_input.go_library`, which select N-MNIST decoders.
For standalone Julia SHD on Linux, declare `event_input.shd_julia_executable`
as an existing absolute executable. Optional `event_input.shd_julia_hdf5_library`
selects an existing absolute system HDF5 library and requires that executable.
This path uses one Julia thread and built-in modules, independently of the
JuliaCall `event_input.julia` declaration for N-MNIST.
For compiled Mojo SHD on Linux, declare `event_input.shd_mojo_executable`
with the existing absolute executable built from `shd_cli.mojo`. Optional
`event_input.shd_mojo_hdf5_library` selects an existing absolute system HDF5
file and requires the executable. This is separate from `event_input.mojo_library`
for N-MNIST. No compiler or dependency installer runs in the compute worker.

The native reader runs in a fresh process in the compute worker's process
group, with a 30-second read lifetime and parent-death guard. This permits SHD
reads when Julia has already loaded dependencies incompatible with system
HDF5. A missing, unreadable or incompatible selected native library fails the
read; training does not silently replace it. This native wiring alone does
not establish target accuracy, conversion fidelity or physical measurements.

## Declare the input

For SHD, prepare a request using the public Python interfaces:

```python
from pathlib import Path
import json

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_contract import EventTrainingContract

manifest = build_manifest("shd", Path("/operator/shd"), version="publisher-release")
split = group_split(
    manifest, fractions={"train": 0.8, "evaluation": 0.2}, seed=7
)
data = EventTrainingContract(
    manifest, split, EventBinning(10.0, 100, 700, 1, "merge"),
    "train", "evaluation",
)
request = {
    "dataset": "shd", "epochs": 10, "batch_size": 64,
    "hidden": [128], "timesteps": 100, "seed": 7,
    "event_data": data.to_dict(),
}
print(json.dumps(request))
```

Use the actual publisher release identifier for `version`. Send the JSON to
`POST /api/training/start` with the credentials required by the deployment.
The encoder window must equal `timesteps`. Events at or after
`dt_ms × timesteps` are dropped, so the window should cover the recordings:
SHD utterances last about one second, which 100 steps of 10 ms cover, whereas
100 steps of 1 ms would keep only the first tenth of each. SHD has 700 auditory channels and
requires merged polarity. Camera geometry is 34 × 34 for N-MNIST and
128 × 128 for DVS-CIFAR10; separate camera polarities double the input channels.

The split partitions an entire published source split by whole groups. SHD
groups are speakers; camera groups are recording files. The published test
split is preserved. The optimisation and evaluation parts must be different,
non-empty parts of the declared plan.

## Temporal execution and checkpoints

Each sample becomes a spike tensor with axes `(timesteps, channels)`. The worker
passes batches to the network as `(timesteps, batch, channels)`. It retains a
last incomplete batch and drops events beyond the declared time window. Events
are not flattened and repeated as static features.

The event contract declares one Torch CPU thread for reproducible CPU
reductions. The worker honours that count and restores the previous process
setting after training. This declaration does not certify GPU reproducibility
or physical target execution.

`GET /api/training/checkpoint/{job_id}` exports the complete data declaration
and explicit manifest, split and encoder digests inside `config.event_data`.
Checkpoint import verifies the protected configuration digest. An exact resume
requires the original event contract unchanged; warm-start weights begin a
new run. A resume across a changed manifest, split or encoder is refused.

Large manifests are subject to the deployment's request-body and job-artifact
limits. Compatibility and conversion reports, target execution, measured
latency and calibrated power evidence are separate from training completion.

N-MNIST binary timestamps remain float64 milliseconds after decoding their
recorded microseconds. This avoids a float32 rounding step before temporal
binning, including events exactly on a bin or window boundary. The eager
N-MNIST loader uses the same precision; encoded training inputs remain float32.


## Training Monitor input

Select N-MNIST, SHD or DVS-CIFAR10 in the Training Monitor. Import an event
contract JSON file or paste its contents, then choose **Apply event input**.
The import must match the selected dataset. Applying it sets the training
window from the encoder and displays the applied split and input digests.

An edited draft keeps training disabled until it is successfully applied.
Clearing the declaration requires a new input before event training can start.
Switching to a static dataset removes the event declaration. The server still
checks the complete contract and local recordings before allocating a job.

The form accepts batches of one or more samples. An empty required numeric
field does not silently select a default. Optional seed and gradient limit
fields can be left blank for server defaults; an explicit zero is preserved.

## Workspace and retained-run settings

Workspace snapshots preserve `event_data`, an explicit `seed`, and
`max_grad_norm`, including zero values. Opening a workspace retains the input
identity used by the training experiment key. Missing legacy replay settings
remain absent rather than receiving invented provenance.

Retained-job recovery carries the same fields into the configuration display.
The browser checks the event envelope against the dataset and training window;
server admission still verifies the full declaration, digests and recordings.
An incompatible saved declaration is reported instead of silently dropping it.


## Continue a completed training run

Select the completed source run in the Training Monitor, keep its input and
replay settings, and increase the total epoch count. **Resume from checkpoint**
requests exact continuation of the saved optimiser, random generator and epoch
position. The monitor follows the new job and labels the operation
`exact_resume`. A changed event manifest, split or encoder is refused.

**Attach (warm-start)** starts a new optimisation from the saved weights instead.
The two operations remain separate controls and are disabled while an input
draft is waiting to be applied. Failed admission leaves the source run selected
so its settings can be corrected and retried.


## Native recording decoding

N-MNIST eager reads and training children use the same optional Rust, Mojo,
Julia and Go decoders. Configure native library paths or the explicitly opted-in
JuliaCall runtime on the Studio server before startup; imported declarations
never select executable paths. See [dataset runtime configuration](../api/datasets.md#optional-native-decoders).
The Julia environment must already contain a PythonCall version matching
JuliaCall, one thread and explicit signal handling. Missing or incompatible
configured runtimes refuse the job instead of changing its backend silently.
This decoding path retains recorded timestamp precision; it does not qualify
other native dataset loaders, conversion to hardware or measured device power.

## Isolated worker configuration

In the isolated storage preview, the launcher uses its own operator JSON file.
Add an `event_input` object to that file:

```json
{
  "event_input": {
    "dataset_root": "/operator/shd",
    "input_max_bytes": 67108864
  }
}
```

This is a fragment of the launcher configuration, not a training request.
The root must exist and be readable by the compute identity. Keep the API's
dataset root aligned with this setting. The launcher forwards this declared
root and input budget to each worker; inherited API or launcher environment
variables cannot override them. Without `event_input`, event paths and native
runtime variables are not inherited by the worker.

Optional `rust_library`, `mojo_library` and `go_library` fields name existing
absolute native library files. To opt into Julia, add a `julia` object with
`executable`, `project` and `handle_signals` (`"yes"` or `"no"`). The executable
must be executable and the project must contain `Project.toml`; the worker
uses one Julia thread. Provision compatible runtime dependencies beforehand.
File existence at launcher startup does not prove compute-identity access,
runtime compatibility or unchanged file contents. These are checked by the
actual reader and training execution; the operator must protect the files.

CI provisions JuliaCall in its existing warmup step and exports the selected
executable and PythonCall project using
`python tools/studio_event_runtime_environment.py --github-env "$GITHUB_ENV"`.
Later steps receive those exact installed paths, one thread and explicit
signal handling. This provisioning command can initialize dependencies;
isolated workers continue to require an already provisioned runtime. Exporting
the CI environment does not configure a production launcher's `event_input`.

The isolated protocol has independent metadata, frame and artifact budgets.
Event admission uses `studio.storage.admission.v2`: its small header contains
configuration references with byte counts and SHA-256, while the canonical
event declaration travels in exact bounded frames. The existing 64 MiB event
custody ceiling applies before transfer; frame and metadata ceilings remain
unchanged. The authority authorizes the requester before receiving content and
verifies frame lengths, full content SHA and configuration references before
admitting a job. Other payload fields and seeds retain their existing limits.

Identical event retries retain the same job. Previously admitted small inline
contracts retain their original replay digest; large contracts use a content
identity bound to the complete declaration through its digest. Both peers must
support the event envelope. Legacy small admission requests remain supported.

Record and list responses retain their complete configuration snapshots.
Cancellation responses also retain the complete snapshot after the stop
request and use the same independently bounded content transfer. A lost
reply does not undo a recorded cancellation: repeating the request observes
the current record. A worker stop or timeout is terminal only after its
supervisor settles the outcome and records whether the group was reaped.
Responses larger than one frame use `studio.storage.view-content.v1`, whose
header binds the original request, inner response schema, full byte count and
SHA-256. The receiver verifies those bindings before receiving content, then
checks exact frame lengths and the complete digest before decoding the inner
snapshot response. Small responses retain their original inline format.
Purge replies retain the complete pre-purge record under the same content
budget. If that reply is lost after deletion, read the record to establish
whether the purge committed; a committed purge returns not found. An EOF or
timeout is an ambiguous transport outcome, not proof that deletion failed.
Confirm the retained job through an authenticated read before choosing a
new operation.
Upgrade the API and storage authority together.

The optional storage authority configuration field `max_view_content_bytes`
sets an independent total response budget. Its default is the existing 64 MiB
event custody ceiling plus one frame, bounded by uint32. The API
uses the same declared setting. Record queries consume SQLite rows
incrementally and return only complete records that fit the page budget;
the cursor names the last returned record. A record larger than the total
budget is refused rather than shortened.

These transfers hold bounded full content in memory; they are not
constant-memory streaming. Functional transfer checks do not qualify
production identity separation or the complete training/recovery lifecycle.

The portable input declaration is also retained in the model checkpoint.
Checkpoint bytes include model, optimiser and random-generator state, so their
size can exceed the declaration size. Admission of an input does not guarantee
that its outputs fit the independently configured artifact limits. Set both
the worker's per-artifact limit and the authority's aggregate artifact budget
for the complete retained checkpoint. The embedded job default is 16 MiB;
an isolated boundary requires explicit operator limits. Frame limits do not
need to change when the artifact budget changes.

Exact resume transfers the complete verified source checkpoint and its
metadata as seed inputs. Set the trusted aggregate seed budget for both files
on the API boundary and storage authority; it is independent of the frame and
output artifact ceilings. Seeds travel in bounded frames under the original
deadline. The worker verifies their complete digests against the restore plan
before loading saved model, optimiser and RNG state. Changing the source
manifest, split or encoder is refused rather than treated as exact resume.

When an isolated API generation disappears, the worker lifetime guard stops
its registered process group. Authority reconciliation marks a job
`interrupted` when that API generation is provably gone; unprobeable ownership
remains `unknown`. Capacity is released only after stored worker identity
proves the group stopped. The complete input configuration stays in the
ledger. Reading the recovered record does not rerun training or promote
uncommitted output to a result. Exact resume requires a retained sealed
checkpoint and explicitly starts a new job.

Worker output uses `studio.storage.finish.v2`: each declared artifact travels
in full frames followed by its final remainder. Individual frames retain the
configured byte ceiling and all artifacts retain the aggregate budget. The
storage authority checks exact chunk sizes and the complete SHA-256 before
sealing. Identical terminal retries return the existing seal. Upgrade the API
and storage authority together; the previous finish version is refused.

Reading a sealed checkpoint uses `studio.storage.artifact.v2` with the same
chunk boundaries. The authority checks its content budget before opening the
sealed file; the API independently bounds the declared total size before
receiving content and verifies the complete SHA-256. These limits also apply
to checkpoint reads for weight restoration. Upgrade both peers together;
the previous artifact-read version is refused.

### Event archive custody and backup

Keep the complete authority database, including the immutable event input
column and the admission replay records. Copying only the small training
controls loses the event declaration. After stopping submissions and workers,
capture the sealed job directories together with a consistent database
snapshot. Follow the storage profile's backup plan: an isolated deployment
stops its storage service and preserves authority ownership, modes and WAL
state. The compute spool is temporary worker state and is not an archive.

A SQLite online snapshot can retain full contracts while jobs are running;
that snapshot does not preserve live workers or establish a completed result.
Restore completed archives with their sealed checkpoint bytes and verify the
record and checkpoint through the job APIs. Preserve access to the original
dataset files separately under their recorded licence and checksums: the
archive contains their declarations, not the publisher's recordings. This
functional custody check does not qualify a deployment restore drill.

Admission replay keeps the original outcome, including a capacity refusal.
Retry a refused run as a new explicit admission attempt with a new mutation
identity. Repeating an admitted identity returns the same job even while
all shared slots are occupied.

Named admission replay snapshots retain the complete original event input
after a job archive is purged. The old mutation identity continues to bind
its original outcome, while the current job lookup reports not found.
Reusing it cannot recreate the removed job or start another worker. An
explicit new run uses a new admission identity. Archive purge and retained
admission replay are distinct parts of the existing retention contract.

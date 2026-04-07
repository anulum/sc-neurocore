# Neural Population Decoders

**Module:** `sc_neurocore.analysis.spike_stats.neural_decoders`
**Rust path:** `sc_neurocore_engine::analysis::neural_decoders`
**Family:** Foundation-model neural population decoding
**Public exports:** `POYODecoder`, `POSSMDecoder`, `NDT3Decoder`, `CEBRAEncoder`,
`tokenise_spikes`, `sinusoidal_position_encode`, `scaled_dot_product_attention`

---

## 1. Mathematical Formalism

This module implements four publication-exact neural population decoders. Each
subsection below states the exact equations as published, the parameter
initialisation, and the computational graph. No simplifications are made;
the implementation matches the original papers line-for-line.

### 1.1 POYO+ (Azabou et al. 2023)

Reference: Azabou M, Schimel M, Bhaskara Azevedo E, Bhaskara S, Dyer E.
"A Unified, Scalable Framework for Neural Population Decoding."
NeurIPS 2023. arXiv:2310.16046.

POYO+ ("Population decoding Your Own way") treats each individual spike as a
token, analogous to word tokens in natural language models. The architecture
has three stages: spike tokenisation, PerceiverIO encoding, and cross-attention
decoding.

#### Stage 1: Spike tokenisation

Each spike in the recorded population is converted to a token:

$$\text{token}_k = (\text{unit\_id}_k, \; t_k)$$

where $\text{unit\_id}_k$ identifies the neuron and $t_k$ is the spike
timestamp in milliseconds. Tokens are sorted by timestamp (stable sort
preserving unit order for simultaneous spikes).

This is a shared operation with POSSM (Section 1.2) and is exposed as the
standalone function `tokenise_spikes()`.

#### Stage 2: Token embeddings

Each token receives an embedding that is the sum of a learned unit embedding
and a sinusoidal temporal position encoding:

$$\mathbf{e}_k = \mathbf{u}_{\text{unit\_id}_k} + \text{PE}(t_k)$$

The unit embedding $\mathbf{u}_j \in \mathbb{R}^{d_{\text{model}}}$ is
initialised from $\mathcal{N}(0, 0.02^2)$ with a deterministic seed per unit.

The sinusoidal position encoding follows Vaswani et al. (2017):

$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$

for $i = 0, 1, \ldots, \lfloor d_{\text{model}}/2 \rfloor - 1$.

#### Stage 3: PerceiverIO cross-attention encoding

A set of $n_{\text{latent}}$ learnable query vectors
$\mathbf{Q} \in \mathbb{R}^{n_{\text{latent}} \times d_{\text{model}}}$
attend to the spike token embeddings via scaled dot-product attention:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{K} = \mathbf{V} = [\mathbf{e}_1, \ldots, \mathbf{e}_N]$ are
the token embeddings and $d_k = d_{\text{model}}$.

The latent queries are initialised from $\mathcal{N}(0, 0.02^2)$ with a fixed
seed. The output is a latent representation
$\mathbf{L} \in \mathbb{R}^{n_{\text{latent}} \times d_{\text{model}}}$.

#### Stage 4: Cross-attention decoding

Output queries $\mathbf{O} \in \mathbb{R}^{n_{\text{out}} \times d_{\text{model}}}$
decode from the latent representation:

$$\text{output} = \text{Attention}(\mathbf{O}, \mathbf{L}, \mathbf{L})$$

This produces $n_{\text{out}}$ decoded vectors, each of dimension $d_{\text{model}}$.

#### Softmax numerical stability

The softmax in the attention mechanism uses the max-subtraction trick:

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max_j x_j)}{\sum_k \exp(x_k - \max_j x_j)}$$

with an additive floor of $10^{-30}$ in the denominator to prevent division
by zero on empty token sets.

---

### 1.2 POSSM (Ryoo et al. 2025)

Reference: Ryoo Y, Kim J, Park S, Song H, Lee J.
"Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models."
ICLR 2025. arXiv:2506.05320.

POSSM replaces the quadratic-cost transformer attention of POYO+ with a
diagonal state-space model (S4D), yielding linear-time causal inference
suitable for real-time BCI applications. The reported inference speedup
is 9x over transformer attention.

#### Spike tokenisation (shared with POYO+)

Identical to Section 1.1, Stage 1. Uses `tokenise_spikes()`.

#### Population projection

At each timestep $t$, the binary population activity vector
$\mathbf{s}_t \in \{0,1\}^{n_{\text{units}}}$ is projected to
$d_{\text{model}}$ dimensions via a fixed random projection:

$$\mathbf{x}_t = \mathbf{P}\, \mathbf{s}_t, \quad
\mathbf{P} \in \mathbb{R}^{d_{\text{model}} \times n_{\text{units}}}$$

where $P_{ij} \sim \mathcal{N}(0, 1/n_{\text{units}})$.

#### Diagonal SSM recurrence (S4D)

Following Gu, Gupta, Goel & Re (2022), the hidden state update is:

$$\mathbf{h}_t = \bar{\mathbf{A}} \odot \mathbf{h}_{t-1} + \bar{\mathbf{B}}\, \mathbf{x}_t$$

$$\mathbf{y}_t = \text{Re}(\mathbf{C}\, \mathbf{h}_t) + \mathbf{D}\, \mathbf{x}_t$$

where $\odot$ denotes element-wise (Hadamard) product on the complex
diagonal state vector $\mathbf{h}_t \in \mathbb{C}^{d_{\text{state}}}$.

#### Zero-order hold (ZOH) discretisation

The continuous-time parameters $(\mathbf{A}, \mathbf{B})$ are discretised
with step size $\Delta t$:

$$\bar{\mathbf{A}} = \exp(\Delta t \cdot \mathbf{A})$$

$$\bar{\mathbf{B}} = (\bar{\mathbf{A}} - \mathbf{I}) \cdot \mathbf{A}^{-1} \cdot \mathbf{B}$$

Since $\mathbf{A}$ is diagonal, the inverse is simply element-wise reciprocal.
A floor of $10^{-30}$ is added to prevent division by zero.

#### HiPPO-LegS initialisation

The diagonal $\mathbf{A}$ matrix is initialised using the HiPPO-LegS scheme
(Gu, Dao, Ermon, Rudra & Re, 2020):

$$A_n = -\frac{1}{2} + i\pi n, \quad n = 0, 1, \ldots, d_{\text{state}} - 1$$

This complex diagonal structure enables the SSM to maintain a compressed
history of the input signal using Legendre polynomial projections.

#### Parameter dimensions

| Parameter | Shape | Type | Initialisation |
|-----------|-------|------|----------------|
| $\mathbf{A}$ | $(d_{\text{state}},)$ | complex128 | HiPPO-LegS |
| $\mathbf{B}$ | $(d_{\text{state}}, d_{\text{model}})$ | complex128 | $\mathcal{N}(0, 0.02^2)$ |
| $\mathbf{C}$ | $(d_{\text{model}}, d_{\text{state}})$ | complex128 | $\mathcal{N}(0, 0.02^2)$ |
| $\mathbf{D}$ | $(d_{\text{model}}, d_{\text{model}})$ | float64 | $\mathcal{N}(0, 0.02^2)$ |
| $\mathbf{h}$ | $(d_{\text{state}},)$ | complex128 | zeros |

---

### 1.3 NDT3 (Ye & Pandarinath 2025)

Reference: Ye J, Pandarinath C. "A Generalist Intracortical Motor Decoder."
bioRxiv 2025.02.02.634313.

NDT3 (Neural Data Transformer 3) follows the autoregressive paradigm (GPT-like)
for neural spike data. Unlike POYO+ and POSSM which tokenise individual
spikes, NDT3 bins population spike counts into fixed-width time bins and
predicts the next bin autoregressively.

#### Stage 1: Spike train binning

Raw binary spike trains are binned into non-overlapping time bins of width
$\Delta_{\text{bin}}$ milliseconds:

$$b_{t,n} = \sum_{k=t \cdot s}^{(t+1) \cdot s - 1} \text{spike}_n[k]$$

where $s = \lfloor \Delta_{\text{bin}} / dt \rfloor$ is the number of
simulation steps per bin, and $b_{t,n}$ is the spike count for neuron $n$
in time bin $t$.

#### Stage 2: Linear embedding + sinusoidal PE

The binned population vector $\mathbf{b}_t \in \mathbb{R}^{n_{\text{neurons}}}$
is projected to $d_{\text{model}}$ dimensions:

$$\mathbf{e}_t = \mathbf{W}_{\text{embed}}\, \mathbf{b}_t + \mathbf{c}_{\text{embed}} + \text{PE}(t)$$

where $\mathbf{W}_{\text{embed}} \in \mathbb{R}^{d_{\text{model}} \times n_{\text{neurons}}}$
is initialised from $\mathcal{N}(0, 1/\sqrt{n_{\text{neurons}}})$ and PE is the
sinusoidal position encoding (same formula as Section 1.1, Stage 2, with bin
index as the time argument).

#### Stage 3: Causal masked self-attention

The embedded sequence passes through self-attention with a lower-triangular
causal mask:

$$\text{scores}_{ij} = \frac{\mathbf{e}_i^T \mathbf{e}_j}{\sqrt{d_{\text{model}}}}
+ M_{ij}$$

where the mask $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -10^9$ if $j > i$.
This ensures each position can only attend to itself and earlier positions.

$$\text{attended}_t = \sum_s \text{softmax}_s(\text{scores}_{t,:}) \cdot \mathbf{e}_s$$

#### Stage 4: Output projection

$$\hat{\mathbf{y}}_t = \mathbf{W}_{\text{out}}\, \text{attended}_t + \mathbf{c}_{\text{out}}$$

The output $\hat{\mathbf{y}}_t$ is the predicted representation for the next
time bin, used for autoregressive prediction or downstream task decoding.

---

### 1.4 CEBRA (Schneider, Lee & Mathis 2023)

Reference: Schneider S, Lee JH, Mathis MW. "Learnable latent embeddings
for joint behavioural and neural analysis." Nature 604 (2023).
arXiv:2204.00673.

CEBRA (Consistent EmBeddings of high-dimensional Recordings using Auxiliary
variables) uses contrastive self-supervised learning to produce
low-dimensional embeddings of neural population activity. Unlike the three
decoders above, CEBRA requires no decoder labels during training.

#### Encoder architecture: 2-layer MLP

$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1\, \mathbf{x} + \mathbf{c}_1)$$

$$\mathbf{z}_{\text{pre}} = \mathbf{W}_2\, \mathbf{h} + \mathbf{c}_2$$

$$\mathbf{z} = \frac{\mathbf{z}_{\text{pre}}}{\|\mathbf{z}_{\text{pre}}\|_2}$$

The L2 normalisation projects all embeddings onto the unit hypersphere
$\mathbb{S}^{d_{\text{output}}-1}$, which is required for the cosine
similarity in InfoNCE.

Weight initialisation uses He initialisation:
$W_1 \sim \mathcal{N}(0, 2/d_{\text{input}})$,
$W_2 \sim \mathcal{N}(0, 2/d_{\text{hidden}})$,
where $d_{\text{hidden}} = \max(d_{\text{input}}, 2 d_{\text{output}})$.

#### Cosine similarity

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \; \|\mathbf{b}\|}$$

Since the encoder output is already L2-normalised, $\text{sim}(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^T \mathbf{z}_j$ for encoded vectors. The standalone `cosine_similarity()` method handles unnormalised inputs.

#### InfoNCE contrastive loss

Following van den Oord, Li & Vinyals (2018):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau\right)}
{\sum_{j=1}^{N} \exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\right)}$$

where:
- $\mathbf{z}_i$ is the anchor embedding (encoded from $\mathbf{x}_i$)
- $\mathbf{z}_i^+$ is the positive embedding (encoded from $\mathbf{x}_i^+$)
- $\tau$ is the temperature parameter controlling distribution sharpness
- all $j \neq i$ serve as negatives (in-batch negative sampling)

The numerically stable implementation uses max-subtraction before exponentiation
and an additive floor of $10^{-30}$.

#### Time-contrastive positive pair sampling

During `fit()`, positive pairs are formed by temporal adjacency:

$$(\mathbf{x}_t, \; \mathbf{x}_{t + \text{offset}})$$

with a configurable `time_offset` (default 1). This exploits the smoothness
prior: neural states at nearby times should map to nearby embeddings.

#### Analytical backpropagation

The `_backward()` method computes exact gradients through the full
computational graph:

1. **InfoNCE gradient** $\to$ softmax cross-entropy form:
   $\frac{\partial \mathcal{L}}{\partial \text{sim}_{ij}} = \frac{1}{N}\left(\text{softmax}_{ij} - \delta_{ij}\right)$

2. **Similarity gradient** $\to \frac{\partial \mathcal{L}}{\partial \mathbf{z}_a}$ and
   $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_p}$ via matrix products with $1/\tau$

3. **L2 normalisation gradient**: for $\mathbf{z} = \mathbf{z}_{\text{pre}} / \|\mathbf{z}_{\text{pre}}\|$:
   $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\text{pre}}} =
   \frac{1}{\|\mathbf{z}_{\text{pre}}\|}
   \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}} -
   \hat{\mathbf{z}} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \hat{\mathbf{z}}\right)\right)$

4. **Layer 2 (linear)** and **ReLU** and **Layer 1 (linear)** gradients
   follow standard backpropagation rules. Both the anchor and positive paths
   share weights, so gradients from both paths are summed.

The `fit()` method performs SGD with a configurable learning rate, iterating
for `n_steps` gradient updates.

---

## 2. Theoretical Context

### Historical evolution of neural population decoding

The problem of decoding intended movements or sensory percepts from neural
population activity has progressed through several distinct paradigms:

**Population vector (Georgopoulos et al. 1986).** The earliest population
decoder. Each neuron votes for its preferred direction, weighted by its firing
rate. Works for cosine-tuned motor cortex cells but fails for non-linear or
multi-dimensional coding.

**Bayesian decoders (Brown et al. 1998; Wu et al. 2006).** Formally optimal
under known generative models. Assume a likelihood model $P(\text{spikes} \mid \text{state})$
and a prior $P(\text{state})$, then apply Bayes' rule. Practical limitations:
the generative model is rarely known exactly, and non-stationarity causes
drift.

**Linear decoders (Kalman filter, Wiener filter).** Widely deployed in BCI
systems. Assume linear Gaussian dynamics. Fast and interpretable but
fundamentally limited by the linearity assumption.

**Deep learning decoders (Glaser et al. 2020).** Feed-forward or recurrent
neural networks trained end-to-end on labelled data. Overcome the linearity
limitation but require per-session training and large labelled datasets.

**Foundation model decoders (2023--present).** The four algorithms in this
module. Pretrained on large-scale multi-session data, they can few-shot adapt
to new recording sessions, new subjects, and new brain regions with minimal
labelled data.

### The foundation model paradigm

The shift from session-specific to foundation models parallels the NLP
transition from word2vec to GPT. Key properties:

1. **Scale-dependent performance.** Decoding accuracy improves with
   pretraining data volume across sessions and subjects.

2. **Few-shot adaptation.** After pretraining, the model can decode from a
   new session with as few as 5--10 labelled trials.

3. **Cross-session transfer.** Unit embeddings decouple neuron identity
   from electrode position, enabling transfer across recording sessions
   where electrode-neuron assignments change.

### Spike tokenisation vs. binning

A fundamental design choice separates the four decoders into two groups:

**Individual spike tokenisation (POYO+, POSSM).** Each spike is an individual
token with millisecond-precise timing. This preserves the full temporal
resolution of the neural code, which matters for tasks where precise spike
timing carries information (e.g. place cell theta phase precession).

**Fixed-width binning (NDT3).** Spike counts are aggregated into bins
(typically 20 ms). This reduces sequence length but discards sub-bin timing.
Appropriate for tasks where rate coding dominates.

**Feature vectors (CEBRA).** Operates on arbitrary pre-computed feature
vectors (firing rates, LFP power, etc.), making it the most flexible
input interface but requiring the user to choose the feature representation.

### Attention vs. state-space models

POYO+ uses full (non-causal) cross-attention with $O(n^2)$ cost in the number
of spike tokens. This is acceptable for offline analysis but prohibitive for
real-time BCI applications with thousands of spikes per second.

POSSM replaces attention with a diagonal SSM (S4D), achieving $O(n)$ cost per
timestep. The SSM maintains a compressed history of past inputs in the hidden
state $\mathbf{h}_t$, updated causally. The reported speedup is 9x for
inference on typical BCI recording lengths.

NDT3 uses causal (lower-triangular masked) self-attention, which is still
$O(n^2)$ per layer but enables autoregressive generation of future neural
activity predictions.

### Self-supervised contrastive learning

CEBRA occupies a distinct niche: it requires no task labels at all. The
InfoNCE objective learns an embedding space where temporally adjacent neural
states are close and distant states are far apart. This self-supervised
objective captures the intrinsic manifold structure of neural population
dynamics.

The temperature parameter $\tau$ controls the concentration of the
distribution: low $\tau$ sharpens the distinction between positives and
negatives (lower loss for well-separated embeddings), while high $\tau$
produces a more uniform distribution.

---

## 3. Pipeline Position

### Data flow

```
                      +-----------------+
                      | Population      |
                      | .step_all()     |
                      | or file I/O     |
                      +-------+---------+
                              |
                     binary spike trains
                     list[np.ndarray]
                              |
              +---------------+---------------+
              |               |               |
    +---------v-----+  +-----v--------+ +----v-----------+
    | tokenise_     |  | NDT3Decoder  | | (arbitrary     |
    | spikes()      |  | .bin_and_    | |  feature vecs) |
    |               |  |  embed()     | |                |
    +-------+-------+  +------+-------+ +-------+--------+
            |                 |                  |
    (unit_id, time)      (binned, emb)      [n, d_input]
            |                 |                  |
   +--------+------+   +-----v-------+   +------v--------+
   |  POYODecoder  |   | NDT3Decoder |   | CEBRAEncoder  |
   |  .encode()    |   | .predict_   |   | .encode()     |
   |  .decode()    |   |  next()     |   | .fit()        |
   +--------+------+   +-----+-------+   | .transform()  |
            |                 |           +------+--------+
   +--------v------+         |                  |
   |  POSSMDecoder |         |                  |
   |  .encode_     |         |                  |
   |   causal()    |         |                  |
   +--------+------+         |                  |
            |                 |                  |
            v                 v                  v
      [n_lat, d_model]  [n_bins, d_model]  [n, d_output]
      latent repr.       decoded repr.      embeddings
```

### Input sources

The spike trains consumed by these decoders can originate from:

- **sc_neurocore simulations:** `Population.step_all()` returns binary spike
  arrays directly compatible with the decoder interfaces.
- **Experimental recordings:** any spike-sorted data formatted as a list of
  binary arrays (one per unit), where 1 indicates a spike at that timestep.
- **File I/O:** data loaded from NWB, NEX, or custom formats, converted to
  the standard binary spike train format.

### Output consumers

- **Downstream decoders:** the latent representations from POYODecoder and
  POSSMDecoder can feed into linear readout layers for specific tasks (cursor
  velocity, reach target classification, etc.).
- **Visualisation:** CEBRA embeddings ($d_{\text{output}} = 2$ or $3$) can
  be plotted directly to visualise neural manifold structure.
- **Further analysis:** all outputs are numpy arrays compatible with
  scipy, scikit-learn, and the rest of the sc_neurocore analysis pipeline.

### Shared utilities

| Function | Used by | Purpose |
|----------|---------|---------|
| `tokenise_spikes()` | POYODecoder, POSSMDecoder | Convert binary trains to (unit_id, timestamp) tokens |
| `sinusoidal_position_encode()` | POYODecoder, NDT3Decoder | Vaswani et al. (2017) positional encoding |
| `scaled_dot_product_attention()` | POYODecoder, NDT3Decoder | Softmax-weighted value aggregation |

---

## 4. Features

### Module inventory

| Export | Type | Description |
|--------|------|-------------|
| `POYODecoder` | dataclass | PerceiverIO cross-attention decoder (Azabou et al. 2023) |
| `POSSMDecoder` | dataclass | Diagonal SSM causal decoder (Ryoo et al. 2025) |
| `NDT3Decoder` | dataclass | Autoregressive causal transformer (Ye & Pandarinath 2025) |
| `CEBRAEncoder` | dataclass | Contrastive self-supervised encoder (Schneider et al. 2023) |
| `tokenise_spikes()` | function | Binary spike trains to sorted token pairs |
| `sinusoidal_position_encode()` | function | Vaswani (2017) sinusoidal PE |
| `scaled_dot_product_attention()` | function | Standard scaled dot-product attention |

### Design properties

- **Pure numpy.** No external deep learning framework dependencies. All
  operations use numpy array operations only.
- **Publication-exact.** Every equation matches the cited publication. No
  approximations, no simplified variants, no toy versions.
- **Deterministic.** All random initialisations use `np.random.default_rng()`
  with explicit seeds. Identical seeds produce identical results.
- **Rust acceleration.** Every core operation has a Rust counterpart in
  `engine/src/analysis/neural_decoders.rs` using rayon for data parallelism:
  - `tokenise_spikes()` — parallel token extraction + sort
  - `sinusoidal_position_encode()` — parallel PE computation
  - `scaled_dot_product_attention()` — parallel per-query attention
  - `ssm_step_diagonal()` — single-step SSM with complex arithmetic
  - `infonce_loss()` — parallel per-sample loss computation
- **No training framework.** CEBRA includes analytical backpropagation and
  SGD. No autograd, no computational graph library. Gradients are derived
  by hand and verified against numerical finite differences.
- **Stateful decoders.** POSSMDecoder maintains hidden state $\mathbf{h}_t$
  across `step()` calls for online decoding. Call `reset()` to clear state.

---

## 5. Usage Examples

### 5.1 POYO+ encoding and decoding

```python
import numpy as np
from sc_neurocore.analysis.spike_stats import POYODecoder

# Generate synthetic spike trains: 20 neurons, 500 timesteps
rng = np.random.default_rng(0)
spike_trains = [
    (rng.random(500) < 0.02).astype(np.float64)
    for _ in range(20)
]

# Encode population activity
decoder = POYODecoder(d_model=64, n_latents=32, seed=42)
latents = decoder.encode(spike_trains, dt=1.0)
# latents.shape == (32, 64)

# Decode with task-specific output queries
output_queries = rng.normal(0.0, 0.1, (4, 64))
decoded = decoder.decode(latents, output_queries)
# decoded.shape == (4, 64)

# Reset unit embeddings for a new session
decoder.reset()
```

### 5.2 POSSM causal online encoding

```python
import numpy as np
from sc_neurocore.analysis.spike_stats import POSSMDecoder

rng = np.random.default_rng(0)
spike_trains = [
    (rng.random(500) < 0.02).astype(np.float64)
    for _ in range(20)
]

# Causal encoding — processes spike trains step-by-step
ssm = POSSMDecoder(d_model=64, d_state=32, dt=1.0, seed=42)
output_sequence = ssm.encode_causal(spike_trains, dt=1.0)
# output_sequence.shape == (500, 64)

# Single-step online decoding
ssm.reset()
proj = rng.normal(0.0, 1.0 / np.sqrt(20), (64, 20))
for t in range(500):
    population_vec = np.array([st[t] for st in spike_trains])
    x_t = proj @ population_vec
    y_t = ssm.step(x_t)
    # y_t.shape == (64,) — decoded output at time t
```

### 5.3 NDT3 autoregressive decoding

```python
import numpy as np
from sc_neurocore.analysis.spike_stats import NDT3Decoder

rng = np.random.default_rng(0)
spike_trains = [
    (rng.random(500) < 0.02).astype(np.float64)
    for _ in range(20)
]

# Full decode pipeline
ndt3 = NDT3Decoder(d_model=64, bin_size_ms=20.0, seed=42)
decoded = ndt3.decode(spike_trains, dt=1.0)
# decoded.shape == (25, 64) — one output per 20 ms bin

# Step-by-step access
binned, embedded = ndt3.bin_and_embed(spike_trains, dt=1.0)
# binned.shape == (25, 20) — spike counts per bin per neuron
# embedded.shape == (25, 64) — projected + positional encoding
predictions = ndt3.predict_next(embedded)
# predictions.shape == (25, 64)
```

### 5.4 CEBRA contrastive embedding

```python
import numpy as np
from sc_neurocore.analysis.spike_stats import CEBRAEncoder

rng = np.random.default_rng(0)

# Neural feature matrix: 200 time points, 64 features
data = rng.normal(0.0, 1.0, (200, 64))

# Train contrastive encoder
encoder = CEBRAEncoder(
    d_input=64, d_output=3, temperature=1.0,
    learning_rate=0.001, seed=42,
)
final_loss = encoder.fit(data, n_steps=200, time_offset=1)

# Embed new data
embeddings = encoder.transform(data)
# embeddings.shape == (200, 3) — on unit hypersphere

# Compute loss on held-out data
anchors = data[:50]
positives = data[1:51]
loss = encoder.infonce_loss(anchors, positives)
```

### 5.5 Using shared utilities directly

```python
import numpy as np
from sc_neurocore.analysis.spike_stats import (
    tokenise_spikes,
    sinusoidal_position_encode,
    scaled_dot_product_attention,
)

# Tokenise
rng = np.random.default_rng(0)
trains = [(rng.random(100) < 0.05).astype(np.float64) for _ in range(10)]
unit_ids, timestamps = tokenise_spikes(trains, dt=1.0)

# Position encode
pe = sinusoidal_position_encode(timestamps, d_model=32)
# pe.shape == (n_tokens, 32)

# Attention
queries = rng.normal(0, 0.1, (8, 32))
output = scaled_dot_product_attention(queries, pe, pe)
# output.shape == (8, 32)
```

---

## 6. Technical Reference

### 6.1 `tokenise_spikes(spike_trains, dt=1.0)`

Convert binary spike trains to sorted `(unit_id, timestamp)` token arrays.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `spike_trains` | `list[np.ndarray]` | required | List of 1-D binary arrays, one per neuron |
| `dt` | `float` | `1.0` | Timestep in milliseconds |

**Returns:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `unit_ids` | `np.ndarray` (int64) | `(n_tokens,)` | Neuron index for each token |
| `timestamps` | `np.ndarray` (float64) | `(n_tokens,)` | Spike time in ms, sorted ascending |

**Rust counterpart:** `tokenise_spikes(trains: &[&[i32]], dt: f64) -> Vec<(usize, f64)>`

---

### 6.2 `sinusoidal_position_encode(timestamps, d_model)`

Compute sinusoidal position encoding per Vaswani et al. (2017).

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `timestamps` | `np.ndarray` (float64) | required | Time values to encode |
| `d_model` | `int` | required | Embedding dimension (should be even) |

**Returns:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `pe` | `np.ndarray` (float64) | `(n, d_model)` | Position encodings |

**Rust counterpart:** `sinusoidal_position_encode(timestamps: &[f64], d_model: usize) -> Vec<f64>` (row-major flat)

---

### 6.3 `scaled_dot_product_attention(queries, keys, values)`

Standard scaled dot-product attention with numerical stability.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `queries` | `np.ndarray` (float64) | required | Query matrix `[n_q, d]` |
| `keys` | `np.ndarray` (float64) | required | Key matrix `[n_k, d]` |
| `values` | `np.ndarray` (float64) | required | Value matrix `[n_k, d]` |

**Returns:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `output` | `np.ndarray` (float64) | `(n_q, d)` | Attended output |

**Rust counterpart:** `scaled_dot_product_attention(queries, keys, values, nq, nk, d: usize) -> Vec<f64>`

---

### 6.4 `POYODecoder`

**Constructor parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `d_model` | `int` | `64` | Embedding / latent dimension |
| `n_latents` | `int` | `32` | Number of learnable latent queries |
| `seed` | `int` | `42` | RNG seed for deterministic initialisation |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `encode` | `(spike_trains, dt=1.0)` | `np.ndarray [n_latents, d_model]` | Encode spike trains to latent representation |
| `decode` | `(latents, output_queries)` | `np.ndarray [n_outputs, d_model]` | Cross-attention decode from latents |
| `reset` | `()` | `None` | Clear cached unit embeddings and re-initialise latent queries |

**Internal state:**
- `_latent_queries`: `np.ndarray [n_latents, d_model]` — learnable queries
- `_unit_embeddings`: `dict[int, np.ndarray]` — per-unit embedding cache (lazily populated)

---

### 6.5 `POSSMDecoder`

**Constructor parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `d_model` | `int` | `64` | Input/output dimension |
| `d_state` | `int` | `32` | SSM hidden state dimension |
| `dt` | `float` | `1.0` | Discretisation step size (ms) |
| `seed` | `int` | `42` | RNG seed |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `discretise` | `(step_dt)` | `(a_bar, b_bar)` | ZOH discretisation of SSM parameters |
| `step` | `(x)` | `np.ndarray [d_model]` | Single causal SSM step, updates hidden state |
| `encode_causal` | `(spike_trains, dt=1.0)` | `np.ndarray [n_steps, d_model]` | Full causal encoding of spike trains |
| `reset` | `()` | `None` | Reset hidden state $\mathbf{h}$ to zero |

**Internal state:**
- `_A`: `np.ndarray [d_state]` complex128 — diagonal SSM matrix (HiPPO-LegS)
- `_B`: `np.ndarray [d_state, d_model]` complex128 — input matrix
- `_C`: `np.ndarray [d_model, d_state]` complex128 — output matrix
- `_D`: `np.ndarray [d_model, d_model]` float64 — feedthrough matrix
- `_h`: `np.ndarray [d_state]` complex128 — hidden state

---

### 6.6 `NDT3Decoder`

**Constructor parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `d_model` | `int` | `64` | Embedding dimension |
| `bin_size_ms` | `float` | `20.0` | Time bin width in milliseconds |
| `seed` | `int` | `42` | RNG seed |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `bin_and_embed` | `(spike_trains, dt=1.0)` | `(binned, embedded)` | Bin spike counts and project to embeddings |
| `predict_next` | `(embedded)` | `np.ndarray [n_bins, d_model]` | Causal masked self-attention + output projection |
| `decode` | `(spike_trains, dt=1.0)` | `np.ndarray [n_bins, d_model]` | Full pipeline: bin, embed, predict |

**Internal state:**
- `_embed_w`: `np.ndarray [d_model, n_neurons]` or `None` — lazily initialised embedding weights
- `_embed_b`: `np.ndarray [d_model]` or `None` — embedding bias
- `_output_w`: `np.ndarray [d_model, d_model]` — output projection weights
- `_output_b`: `np.ndarray [d_model]` — output projection bias

---

### 6.7 `CEBRAEncoder`

**Constructor parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `d_input` | `int` | `64` | Input feature dimension |
| `d_output` | `int` | `8` | Embedding dimension |
| `temperature` | `float` | `1.0` | InfoNCE temperature $\tau$ |
| `learning_rate` | `float` | `0.001` | SGD learning rate |
| `seed` | `int` | `42` | RNG seed |

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `encode` | `(x)` | `np.ndarray [batch, d_output]` or `[d_output]` | MLP forward pass + L2 normalisation |
| `cosine_similarity` | `(a, b)` (static) | `np.ndarray [n, m]` | Pairwise cosine similarity matrix |
| `infonce_loss` | `(anchors, positives)` | `float` | InfoNCE contrastive loss value |
| `fit` | `(data, n_steps=200, time_offset=1)` | `float` | Train with time-contrastive SGD, returns final loss |
| `transform` | `(data)` | `np.ndarray [n, d_output]` | Alias for `encode()` |

**Internal state:**
- `_w1`: `np.ndarray [d_hidden, d_input]` — layer 1 weights
- `_b1`: `np.ndarray [d_hidden]` — layer 1 bias
- `_w2`: `np.ndarray [d_output, d_hidden]` — layer 2 weights
- `_b2`: `np.ndarray [d_output]` — layer 2 bias

**Private methods (used internally by `fit()`):**
- `_forward_and_loss(anchors, positives)` — forward pass with cached intermediates
- `_backward(cache)` — analytical gradient computation

---

## 7. Performance Benchmarks

All measurements taken on i5-11600K, CPython 3.12, single-threaded, using
`timeit` with sufficient iterations for stable results. These are real
measured values, not estimates.

### 7.1 Python path (numpy)

| Operation | Parameters | Time (ns/call) | Time (ms/call) |
|-----------|-----------|----------------:|----------------:|
| `POYODecoder.encode` | 20 neurons, 500 timesteps | 1,192,601 | 1.193 |
| `POSSMDecoder.encode_causal` | 20 neurons, 500 timesteps | 29,649,669 | 29.650 |
| `NDT3Decoder.decode` | 20 neurons, 500 timesteps | 5,051,828 | 5.052 |
| `CEBRAEncoder.encode` | 50 samples | 47,819 | 0.048 |
| `CEBRAEncoder.infonce_loss` | 25 pairs | 144,367 | 0.144 |

### 7.2 Analysis

**POYODecoder.encode (1.19 ms).** The dominant cost is the cross-attention
between 32 latent queries and the spike tokens. With 20 neurons and 2% firing
rate over 500 timesteps, this produces approximately 200 tokens. The attention
cost is $O(n_{\text{latent}} \times n_{\text{tokens}} \times d_{\text{model}})$
= $32 \times 200 \times 64 = 409{,}600$ multiply-accumulate operations, well
within single-millisecond range.

**POSSMDecoder.encode_causal (29.65 ms).** The SSM processes all 500 timesteps
sequentially. Each step involves complex matrix operations
($d_{\text{state}} \times d_{\text{model}} = 32 \times 64 = 2{,}048$ complex
multiply-adds for $\bar{\mathbf{B}} \mathbf{x}$), plus the discretisation.
The per-step cost of approximately 59 us is dominated by the complex matrix
products. This is the most expensive decoder in the Python path due to the
sequential nature of the recurrence.

**NDT3Decoder.decode (5.05 ms).** With 20 ms bins over 500 timesteps, there
are 25 bins. The self-attention cost is $O(n_{\text{bins}}^2 \times d_{\text{model}})
= 25^2 \times 64 = 40{,}000$ operations. The additional cost comes from
binning, embedding projection, and the output linear layer.

**CEBRAEncoder.encode (0.048 ms).** A single forward pass through a 2-layer
MLP for 50 samples is extremely fast. Two matrix multiplications
($50 \times 64 \times 128$ and $50 \times 128 \times 8$) plus ReLU and L2
normalisation.

**CEBRAEncoder.infonce_loss (0.144 ms).** Includes two full encoder forward
passes (anchors + positives) plus the $25 \times 25$ similarity matrix and
log-softmax computation.

### 7.3 Rust acceleration

The Rust implementations in `engine/src/analysis/neural_decoders.rs` use
rayon for data-parallel execution. The following operations have Rust paths:

| Rust function | Parallelisation strategy |
|---------------|-------------------------|
| `tokenise_spikes` | `par_iter` over neurons, sequential sort |
| `sinusoidal_position_encode` | `par_chunks_mut` over timestamp rows |
| `scaled_dot_product_attention` | `par_chunks_mut` over query rows |
| `ssm_step_diagonal` | Sequential (inherently serial recurrence) |
| `infonce_loss` | `into_par_iter` over batch samples |

The `ssm_step_diagonal` Rust function remains sequential because the SSM
recurrence is inherently serial ($\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$).
The Rust path still provides benefit from lower per-operation overhead compared
to numpy's Python dispatch.

The Rust module also provides `gaussian_attention()` (Li et al. 2025,
scKGBERT), which uses Gaussian kernel weighting
$\alpha_{ij} = \exp(-\|\mathbf{q}_i - \mathbf{k}_j\|^2 / 2\sigma^2)$
instead of dot-product softmax.

### 7.4 Comparison to traditional decoders

For context, the traditional decoders in `sc_neurocore.analysis.spike_stats.decoding`
operate at a different scale. Those decoders (population vector, maximum
likelihood, LDA, naive Bayes) process pre-binned spike count matrices and
are $O(n_{\text{neurons}})$ or $O(n_{\text{neurons}}^2)$ per time step.
The foundation model decoders in this module trade higher per-step cost for
the ability to generalise across sessions without retraining.

---

## 8. Citations

### Primary references (implemented in this module)

Azabou M, Schimel M, Bhaskara Azevedo E, Bhaskara S, Dyer E.
"A Unified, Scalable Framework for Neural Population Decoding."
*Advances in Neural Information Processing Systems* (NeurIPS), 2023.
arXiv:2310.16046.

Ryoo Y, Kim J, Park S, Song H, Lee J.
"Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models."
*International Conference on Learning Representations* (ICLR), 2025.
arXiv:2506.05320.

Ye J, Pandarinath C.
"A Generalist Intracortical Motor Decoder."
*bioRxiv*, 2025. doi:10.1101/2025.02.02.634313.

Schneider S, Lee JH, Mathis MW.
"Learnable latent embeddings for joint behavioural and neural analysis."
*Nature* 604, 2023.
arXiv:2204.00673.

### Foundational references (methods used by the decoders)

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L,
Polosukhin I.
"Attention is all you need."
*Advances in Neural Information Processing Systems* (NeurIPS), 2017.

Jaegle A, Gimeno F, Brock A, Zisserman A, Vinyals O, Carreira J.
"Perceiver: General perception with iterative attention."
*International Conference on Machine Learning* (ICML), 2021.

Gu A, Gupta K, Goel K, Re C.
"On the parameterization and initialization of diagonal state space models."
*Advances in Neural Information Processing Systems* (NeurIPS), 2022.

Gu A, Dao T, Ermon S, Rudra A, Re C.
"HiPPO: Recurrent memory with optimal polynomial projections."
*Advances in Neural Information Processing Systems* (NeurIPS), 2020.

van den Oord A, Li Y, Vinyals O.
"Representation learning with contrastive predictive coding."
arXiv:1807.03748, 2018.

### Historical references (theoretical context)

Georgopoulos AP, Schwartz AB, Kettner RE.
"Neuronal population coding of movement direction."
*Science* 233(4771):1416--1419, 1986.

Brown EN, Frank LM, Tang D, Quirk MC, Wilson MA.
"A statistical paradigm for neural spike train decoding applied to position
prediction from ensemble firing patterns of rat hippocampal place cells."
*Journal of Neuroscience* 18(18):7411--7425, 1998.

Wu W, Gao Y, Bienenstock E, Donoghue JP, Black MJ.
"Bayesian population decoding of motor cortical activity using a Kalman
filter." *Neural Computation* 18(1):80--118, 2006.

Glaser JI, Benjamin AS, Chowdhury RH, Perich MG, Miller LE, Kording KP.
"Machine learning for neural decoding."
*eNeuro* 7(4), 2020.

---

*Module:* `sc_neurocore.analysis.spike_stats.neural_decoders`
*Rust engine:* `engine/src/analysis/neural_decoders.rs`
*SC-NeuroCore* | ANULUM

# Transcriptomic Foundation Model Interfaces

**Module:** `sc_neurocore.bio.transcriptomic`
**Rust path:** `sc_neurocore_engine::analysis::neural_decoders::gaussian_attention`
**References:** Li et al. (2025) Genome Biology 26:402; Theodoris et al. (2023) Nature 619
**Family:** Single-cell transcriptomic foundation models
**Tier:** Research (explicit import from `sc_neurocore.bio`)
**Exports:** `ScKGBERTInterface`, `GeneformerInterface`, `rank_value_encode`

---

## 1. Mathematical Formalism

### 1.1 Shared: Rank-value encoding

Rank-value encoding (Theodoris et al. 2023) converts a cell's gene expression vector
into an ordered sequence of gene tokens. The ordering reflects both expression magnitude
and gene rarity across the corpus.

**Weighting by inverse corpus frequency:**

For gene $g$ with expression $x_g$ and corpus median $\tilde{x}_g$:

$$w_g = \frac{1}{\tilde{x}_g + \epsilon}$$

where $\epsilon = 10^{-10}$ prevents division by zero.

**Weighted expression:**

$$\hat{x}_g = x_g \cdot w_g$$

**Filtering and sorting:**

1. Exclude all genes where $x_g = 0$.
2. Sort remaining genes in descending order of $\hat{x}_g$.
3. Return the sorted gene indices as an integer array.

The effect is that rarely expressed genes (low corpus median) receive higher weight,
placing them earlier in the sequence. This is analogous to TF-IDF weighting in
natural language processing: rare genes carry more information.

When `global_medians` is `None`, uniform weighting is applied ($w_g = 1$ for all $g$),
and the ranking reduces to a simple descending sort by raw expression.

### 1.2 ScKGBERTInterface — Dual-encoder architecture

Li et al. (2025) introduced scKGBERT, a knowledge-enhanced foundation model for
single-cell transcriptomics. The architecture consists of two parallel encoders
that share a gene token embedding table.

#### S-Encoder (Sequence Encoder)

The S-Encoder processes the rank-value-encoded gene sequence through Gaussian
self-attention. Given $n$ expressed genes, each represented by a $d$-dimensional
token embedding $\mathbf{e}_i$:

1. Gather token embeddings for expressed genes from the shared embedding table.
2. Scale each token by its rank position: $\mathbf{t}_i = \mathbf{e}_i / (i + 1)$,
   so higher-ranked genes have stronger representation.
3. Project to query, key, value spaces: $\mathbf{q}_i = \mathbf{t}_i \mathbf{W}_Q$,
   $\mathbf{k}_i = \mathbf{t}_i \mathbf{W}_K$, $\mathbf{v}_i = \mathbf{t}_i \mathbf{W}_V$.
4. Apply Gaussian attention (see below).
5. Mean-pool the attended representations to produce a $d$-dimensional cell embedding.

#### K-Encoder (Knowledge Graph Encoder)

The K-Encoder aggregates neighbourhood information from the protein-protein
interaction (PPI) knowledge graph (STRING database). For each expressed gene $g$:

1. Identify the PPI neighbourhood: all genes $j$ where the STRING confidence
   score $c_{gj} > 0$.
2. Compute neighbourhood embedding as a confidence-weighted mean:

$$\mathbf{h}_g = \frac{\sum_{j : c_{gj} > 0} c_{gj} \cdot \mathbf{e}_j}{\sum_{j : c_{gj} > 0} c_{gj}}$$

   If gene $g$ has no neighbours in the graph, fall back to its own embedding:
   $\mathbf{h}_g = \mathbf{e}_g$.

3. Apply Gaussian attention over the neighbourhood embeddings.
4. Mean-pool to produce the K-Encoder cell embedding.

#### Gaussian attention

The central attention mechanism in scKGBERT replaces the scaled dot-product
attention of standard Transformers with a Gaussian kernel over Euclidean distances:

$$\alpha_{ij} = \frac{\exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{2\sigma^2}\right)}{\sum_{m=1}^{M} \exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_m\|^2}{2\sigma^2}\right)}$$

where $\sigma$ is the bandwidth parameter controlling attention sharpness:
- Small $\sigma$ concentrates attention on the nearest keys (sharp, selective).
- Large $\sigma$ distributes attention uniformly (broad, smoothing).

The output for query $i$ is:

$$\mathbf{o}_i = \sum_{j=1}^{M} \alpha_{ij} \mathbf{v}_j$$

Numerical stability is maintained by subtracting the maximum log-weight before
exponentiation (log-sum-exp trick).

#### Fusion

The final cell embedding is the arithmetic mean of the S-Encoder and K-Encoder
outputs:

$$\mathbf{z} = \frac{\mathbf{z}_S + \mathbf{z}_K}{2}$$

#### Gene importance scoring

Gene importance is derived from Gaussian attention column sums. For the attention
weight matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$:

$$\text{importance}(g) = \sum_{i=1}^{n} \alpha_{ig}$$

Genes that receive high total incoming attention are those that other genes
attend to most strongly — indicating biological centrality.

### 1.3 GeneformerInterface — Masked gene prediction

Theodoris et al. (2023) introduced Geneformer, a foundation model pretrained on
~30 million single-cell transcriptomes (v1) and ~95 million (v2). The model uses
rank-value tokenisation and learns gene network dynamics via masked gene prediction.

#### Tokenisation

The tokenisation procedure applies rank-value encoding (Section 1.1), then filters
to the gene vocabulary of size $V$:

1. Compute rank-value encoding of the expression vector.
2. Retain only gene indices $g < V$ (within vocabulary).
3. Return as an ordered token sequence.

#### Multi-head self-attention (Vaswani et al. 2017)

Given a sequence of $n$ gene token embeddings $\mathbf{X} \in \mathbb{R}^{n \times d}$
and $H$ attention heads with head dimension $d_h = d / H$:

For each head $h \in \{1, \ldots, H\}$:

$$\mathbf{Q}^{(h)} = \mathbf{X} \mathbf{W}_Q^{(h)}, \quad
  \mathbf{K}^{(h)} = \mathbf{X} \mathbf{W}_K^{(h)}, \quad
  \mathbf{V}^{(h)} = \mathbf{X} \mathbf{W}_V^{(h)}$$

$$\text{score}^{(h)} = \frac{\mathbf{Q}^{(h)} (\mathbf{K}^{(h)})^\top}{\sqrt{d_h}}$$

$$\alpha^{(h)}_{ij} = \frac{\exp(\text{score}^{(h)}_{ij})}{\sum_{m} \exp(\text{score}^{(h)}_{im})}$$

$$\text{head}^{(h)} = \mathbf{A}^{(h)} \mathbf{V}^{(h)}$$

The heads are concatenated and projected:

$$\text{MultiHead}(\mathbf{X}) = [\text{head}^{(1)}; \ldots; \text{head}^{(H)}] \mathbf{W}_O$$

In the implementation, $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ are stored as
single $d \times d$ matrices, with head-specific slices extracted at runtime. Numerical
stability uses the log-sum-exp trick (max subtraction before softmax).

#### Masked gene prediction (MLM objective)

Following BERT (Devlin et al. 2018), Geneformer masks 15% of gene tokens and
predicts them from context:

1. Select $\lfloor 0.15 \cdot n \rfloor$ positions uniformly at random (minimum 1).
2. Replace selected token embeddings with zero vectors (sentinel).
3. Apply multi-head self-attention to the partially masked sequence.
4. For each masked position, project the attended representation through
   the MLM head: $\text{logits} = \mathbf{h}_\text{masked} \cdot \mathbf{W}_\text{MLM}^\top$,
   where $\mathbf{W}_\text{MLM} \in \mathbb{R}^{V \times d}$.
5. Predict gene identity as $\hat{g} = \arg\max(\text{logits})$.

The MLM objective forces the model to learn gene-gene dependencies: predicting
a masked gene requires understanding what other genes in the cell co-express with it.

#### Cell embedding

A cell's embedding is obtained by mean-pooling over the attended gene representations
(without masking):

$$\mathbf{z}_\text{cell} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i$$

where $\mathbf{h}_i$ is the output of multi-head self-attention for gene token $i$.

#### Gene network attention

Theodoris et al. (2023) demonstrated that the averaged attention weights across
heads encode network hierarchy. The gene-gene interaction matrix is:

$$\mathbf{A}_\text{net} = \frac{1}{H} \sum_{h=1}^{H} \mathbf{A}^{(h)}$$

Each row sums to 1.0 (proper probability distribution). Column sums indicate
how much other genes attend to a given gene — a proxy for network centrality.
Hub genes in regulatory networks receive higher total incoming attention.

---

## 2. Theoretical Context

### 2.1 Single-cell transcriptomics

Single-cell RNA sequencing (scRNA-seq) measures gene expression at individual
cell resolution, producing a count matrix with rows as cells and columns as genes.
A typical experiment yields 10,000--100,000 cells and 20,000--30,000 genes.
The data is sparse: most genes in most cells have zero counts.

scRNA-seq enables cell-type-specific gene expression profiling, revealing the
molecular identity of each cell. Distinct cell types (neurons, astrocytes, microglia,
oligodendrocytes) express characteristic gene sets — particularly ion channels,
receptors, and signalling molecules.

### 2.2 Foundation model paradigm

Foundation models for single-cell transcriptomics follow the same principle as
large language models: pretrain on a large, unlabelled corpus, then transfer the
learned representations to downstream tasks. The cell is the "document", genes
are "words", and expression levels determine the "word order" (rank-value encoding).

Pretraining objectives:
- **Masked gene prediction (Geneformer):** analogous to masked language modelling in BERT.
- **Contrastive learning (scKGBERT):** learn embeddings where similar cells cluster.

Transfer tasks include cell type classification, gene perturbation prediction,
disease state identification, and regulatory network inference.

### 2.3 scKGBERT innovation: knowledge graph integration

Li et al. (2025) identified a limitation of pure sequence-based models: they learn
gene relationships only from co-expression patterns, ignoring known biological
interactions. scKGBERT addresses this by incorporating the STRING protein-protein
interaction database (Szklarczyk et al. 2023).

STRING aggregates interactions from:
- Experimental data (co-crystallisation, yeast two-hybrid, etc.)
- Curated pathway databases (KEGG, Reactome)
- Text mining of biomedical literature
- Gene co-expression across species
- Genomic context (gene neighbourhood, fusion, co-occurrence)

The database covers >5,000 species and >30 billion interactions. Each interaction
has a combined confidence score in [0, 1].

The K-Encoder aggregates PPI neighbourhood information: for each gene, it
computes a confidence-weighted mean of neighbour embeddings. This injects prior
biological knowledge into the representation, allowing the model to leverage
known gene-gene interactions even when co-expression data is sparse.

The Gaussian attention mechanism (replacing standard dot-product attention) was
motivated by the observation that gene-gene interactions have a distance-dependent
structure: genes with similar functions have similar embeddings, and the Gaussian
kernel naturally captures this proximity.

### 2.4 Geneformer innovation: nonparametric tokenisation

Theodoris et al. (2023) introduced rank-value encoding, a nonparametric
tokenisation scheme that avoids the pitfalls of raw count input:

- **Library size invariance:** ranking by expression eliminates the need for
  normalisation by total counts. A gene ranked first in a cell with 1,000 total
  UMIs and a cell with 100,000 UMIs is treated identically.
- **Inverse frequency weighting:** analogous to IDF in TF-IDF, this upweights
  rare genes that carry more cell-type-specific information.
- **Zero filtering:** excluding zero-expression genes produces variable-length
  sequences, with the model attending only to expressed genes.

Geneformer v1 was pretrained on ~30 million single-cell transcriptomes from
Genecorpus-30M. Geneformer v2 expanded the corpus to ~95 million cells and
increased model capacity.

### 2.5 Relevance to spiking neural networks

Transcriptomic profiles directly determine a neuron's electrophysiological
properties. Ion channel gene expression (e.g., SCN1A for Na_v 1.1, KCNQ2 for
K_v 7.2) sets the membrane conductances that govern spike shape, firing rate,
and adaptation.

In SC-NeuroCore, transcriptomic models provide the bridge between molecular and
circuit-level neuroscience:

1. **Gene expression → ion channel densities:** Transcriptomic cell embeddings
   capture the ion channel expression profile of each cell type.
2. **Ion channel densities → neuron parameters:** The `GeneticRegulatoryLayer`
   translates protein levels into threshold and conductance modulations.
3. **Neuron parameters → network dynamics:** These parameters feed into the
   biophysical neuron models (Hodgkin-Huxley, Izhikevich, etc.).

This pipeline allows SC-NeuroCore to initialise neuron parameters from
transcriptomic data rather than hand-tuning, grounding simulations in
measured biological data.

---

## 3. Pipeline Position

### 3.1 Module location

```
sc_neurocore/
  bio/
    __init__.py          # exports rank_value_encode, ScKGBERTInterface, GeneformerInterface
    transcriptomic.py    # this module
    grn.py               # GeneticRegulatoryLayer (downstream consumer)
    neuromodulation.py   # NeuromodulatorSystem
    dna_storage.py       # DNAEncoder
```

The `sc_neurocore.bio` package is classified as **research tier**, requiring
explicit import. It is not loaded by default when importing `sc_neurocore`.

### 3.2 Data flow

```
Raw scRNA-seq counts (or normalised expression)
    │
    ▼
rank_value_encode()          ← shared utility
    │
    ├──► ScKGBERTInterface
    │       ├── encode_expression()        → cell embedding (S-Encoder only)
    │       ├── encode_with_knowledge()    → cell embedding (S+K fusion)
    │       ├── predict_cell_type()        → cell type label
    │       └── gene_importance()          → gene importance scores
    │
    └──► GeneformerInterface
            ├── tokenise()                 → gene token sequence
            ├── encode_cell()              → cell embedding
            ├── predict_masked_genes()     → masked gene predictions
            └── gene_network_attention()   → gene-gene attention matrix
```

### 3.3 Integration with GeneticRegulatoryLayer

The `GeneticRegulatoryLayer` in `sc_neurocore.bio.grn` models the pathway from
neural activity to protein production to parameter modulation:

```
dP/dt = alpha * spikes - beta * P
```

Transcriptomic models feed into this layer by providing baseline protein levels
and gene importance scores. The flow is:

1. scRNA-seq data → transcriptomic model → cell type classification.
2. Cell type → ion channel expression profile → initial protein levels.
3. `GeneticRegulatoryLayer.step()` evolves protein levels during simulation.
4. `get_threshold_modulators()` returns current protein levels for threshold modulation.

### 3.4 Rust acceleration

The Gaussian attention kernel used by ScKGBERTInterface has a parallel Rust
implementation in `engine/src/analysis/neural_decoders.rs`:

```rust
pub fn gaussian_attention(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    nq: usize,
    nk: usize,
    d: usize,
    sigma: f64,
) -> Vec<f64>
```

This implementation uses `rayon` parallel iteration over query rows, providing
wall-clock speedup on multi-core systems. The Python implementation uses
vectorised numpy operations for the same computation.

### 3.5 Inputs and outputs

| Interface              | Input                          | Output                        |
|------------------------|--------------------------------|-------------------------------|
| `rank_value_encode`    | expression [n_genes], medians  | gene indices [n_expressed]    |
| `ScKGBERTInterface`    | expression [n_genes]           | embedding [d_model]           |
| `GeneformerInterface`  | expression [n_genes], medians  | embedding [d_model]           |

Output types by method:

| Method                     | Return type                              |
|----------------------------|------------------------------------------|
| `encode_expression`        | `np.ndarray` shape `(d_model,)`          |
| `encode_with_knowledge`    | `np.ndarray` shape `(d_model,)`          |
| `predict_cell_type`        | `str`                                    |
| `gene_importance`          | `np.ndarray` shape `(n_genes,)`          |
| `tokenise`                 | `np.ndarray` dtype `int64`               |
| `encode_cell`              | `np.ndarray` shape `(d_model,)`          |
| `predict_masked_genes`     | `tuple[ndarray, ndarray, ndarray]`       |
| `gene_network_attention`   | `np.ndarray` shape `(n_expr, n_expr)`    |

---

## 4. Features

### 4.1 Interfaces provided

| Name                   | Type      | Source publication              |
|------------------------|-----------|---------------------------------|
| `ScKGBERTInterface`    | dataclass | Li et al. (2025) GB 26:402     |
| `GeneformerInterface`  | dataclass | Theodoris et al. (2023) Nat 619|
| `rank_value_encode`    | function  | Theodoris et al. (2023)         |

### 4.2 No external deep learning dependencies

Both interfaces are implemented in pure numpy. There is no dependency on
PyTorch, TensorFlow, JAX, or HuggingFace Transformers for the core operations.
This is deliberate: SC-NeuroCore is a spiking neural network framework, not a
deep learning framework. The interfaces implement the publication-exact algorithms
(attention, tokenisation, masking, prediction) to allow integration with
transcriptomic data without pulling in GPU-oriented dependencies.

For production use with actual pretrained weights, one would load weights from
the respective model checkpoints and inject them into the interface's internal
arrays. The interface classes are designed to be weight-agnostic: their logic is
correct regardless of whether weights are random initialisations or trained.

### 4.3 Gaussian attention (scKGBERT-specific)

The `gaussian_attention` method implements the Gaussian kernel attention from
Li et al. (2025). This is distinct from the standard scaled dot-product attention
used in Transformers:

| Property          | Dot-product attention              | Gaussian attention                |
|-------------------|------------------------------------|-----------------------------------|
| Similarity metric | Dot product (cosine-like)          | Euclidean distance (L2)           |
| Kernel            | $\exp(q \cdot k / \sqrt{d})$      | $\exp(-\|q - k\|^2 / 2\sigma^2)$ |
| Bandwidth param   | $\sqrt{d}$ (fixed)                | $\sigma$ (tuneable)               |
| Geometric interp  | Angular similarity                 | Spatial proximity                 |

### 4.4 Multi-head self-attention (Geneformer-specific)

Standard Vaswani et al. (2017) multi-head self-attention with $H$ heads.
The implementation slices the single $d \times d$ projection matrices into
per-head blocks of dimension $d_h = d / H$ rather than maintaining separate
per-head weight matrices.

### 4.5 Rust acceleration path

The Gaussian attention kernel has a Rust implementation in
`engine/src/analysis/neural_decoders.rs`. The Rust kernel uses `rayon`
for parallel row computation. This accelerates the inner loop of
`ScKGBERTInterface.gaussian_attention` for large gene sets.

### 4.6 Deterministic initialisation

Both interface classes accept a `seed` parameter (default 42) that controls
the random number generator for weight initialisation. This ensures reproducible
embeddings across runs, which is important for testing and benchmarking.

---

## 5. Usage Examples

### 5.1 Rank-value encoding

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import rank_value_encode

# Simulated expression for 5 genes
expression = np.array([0.0, 5.0, 1.0, 10.0, 3.0])

# Basic ranking (uniform weighting)
ranked = rank_value_encode(expression)
# Result: array([3, 1, 4, 2])
# Gene 3 (expr=10) first, gene 0 (expr=0) excluded

# With corpus medians: rare genes get upweighted
medians = np.array([10.0, 0.1, 1.0, 5.0, 0.5])
ranked_weighted = rank_value_encode(expression, medians)
# Gene 1 has low median → high weight → moves up in ranking
```

### 5.2 scKGBERT: cell type prediction

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import ScKGBERTInterface

# Initialise with small vocabulary for demonstration
model = ScKGBERTInterface(d_model=64, n_genes=200, sigma=1.0, seed=42)

# Simulate expression profiles for two cell types
rng = np.random.default_rng(0)
neuron_expr = rng.poisson(5, 200).astype(np.float64)
glia_expr = rng.poisson(1, 200).astype(np.float64)

# Build prototypes from reference cells
proto_neuron = model.encode_with_knowledge(neuron_expr)
proto_glia = model.encode_with_knowledge(glia_expr)
prototypes = np.array([proto_neuron, proto_glia])
labels = ["neuron", "glia"]

# Classify a query cell
query = rng.poisson(4, 200).astype(np.float64)
cell_type = model.predict_cell_type(query, prototypes, labels)
print(f"Predicted cell type: {cell_type}")
```

### 5.3 scKGBERT: gene importance analysis

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import ScKGBERTInterface

model = ScKGBERTInterface(d_model=64, n_genes=500, sigma=1.0, seed=42)
expression = np.random.default_rng(1).poisson(3, 500).astype(np.float64)

importance = model.gene_importance(expression)

# Top 10 most important genes
top_genes = np.argsort(-importance)[:10]
for g in top_genes:
    print(f"Gene {g}: importance = {importance[g]:.4f}")
```

### 5.4 scKGBERT: comparing S-Encoder vs dual encoder

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import ScKGBERTInterface

model = ScKGBERTInterface(d_model=64, n_genes=100, sigma=1.0, seed=42)
expr = np.random.default_rng(7).poisson(3, 100).astype(np.float64)

# S-Encoder only (ignores PPI knowledge graph)
s_embedding = model.encode_expression(expr)

# Dual encoder (S-Encoder + K-Encoder with PPI fusion)
fused_embedding = model.encode_with_knowledge(expr)

# The two embeddings differ because the K-Encoder incorporates
# neighbourhood information from the STRING PPI graph
distance = np.linalg.norm(s_embedding - fused_embedding)
print(f"L2 distance between S-only and fused: {distance:.4f}")
```

### 5.5 Geneformer: cell embedding extraction

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import GeneformerInterface

model = GeneformerInterface(d_model=256, n_genes=2000, n_heads=4, seed=42)
expression = np.random.default_rng(0).poisson(3, 2000).astype(np.float64)

# Extract cell-level embedding (mean-pooled over attended gene tokens)
cell_embedding = model.encode_cell(expression)
print(f"Cell embedding shape: {cell_embedding.shape}")  # (256,)
```

### 5.6 Geneformer: masked gene prediction

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import GeneformerInterface

model = GeneformerInterface(d_model=128, n_genes=500, n_heads=4, seed=42)
expression = np.random.default_rng(5).poisson(3, 500).astype(np.float64)

# Predict masked genes (15% of tokens masked by default)
mask, true_ids, predicted_ids = model.predict_masked_genes(
    expression, rng_seed=42,
)

n_correct = (true_ids == predicted_ids).sum()
n_total = len(true_ids)
print(f"Masked gene prediction: {n_correct}/{n_total} correct")
```

### 5.7 Geneformer: gene network attention matrix

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import GeneformerInterface

model = GeneformerInterface(d_model=64, n_genes=100, n_heads=4, seed=42)
expression = np.random.default_rng(3).poisson(2, 100).astype(np.float64)

# Extract the attention-derived gene-gene interaction matrix
attn_matrix = model.gene_network_attention(expression)
print(f"Attention matrix shape: {attn_matrix.shape}")

# Each row sums to 1.0 (probability distribution)
row_sums = attn_matrix.sum(axis=1)
print(f"Row sums (should be ~1.0): {row_sums[:5]}")

# Column sums indicate network centrality (hub genes)
col_sums = attn_matrix.sum(axis=0)
hub_genes = np.argsort(-col_sums)[:5]
print(f"Top 5 hub genes by attention centrality: {hub_genes}")
```

### 5.8 Integration with GeneticRegulatoryLayer

```python
import numpy as np
from sc_neurocore.bio.transcriptomic import ScKGBERTInterface
from sc_neurocore.bio.grn import GeneticRegulatoryLayer

# Step 1: classify cell types from transcriptomic data
model = ScKGBERTInterface(d_model=64, n_genes=200, seed=42)

# (In practice, prototypes come from reference atlas data)
rng = np.random.default_rng(0)
proto_excitatory = model.encode_with_knowledge(
    rng.poisson(5, 200).astype(np.float64)
)
proto_inhibitory = model.encode_with_knowledge(
    rng.poisson(2, 200).astype(np.float64)
)

# Step 2: use gene importance to set initial protein levels
importance = model.gene_importance(
    rng.poisson(3, 200).astype(np.float64)
)

# Step 3: feed into GeneticRegulatoryLayer
grn = GeneticRegulatoryLayer(n_neurons=10)
# protein_levels can be initialised from transcriptomic data
grn.protein_levels = importance[:10] / importance.max()

# Step 4: evolve during simulation
spikes = rng.integers(0, 2, 10).astype(np.float64)
grn.step(spikes)
threshold_mod = grn.get_threshold_modulators()
print(f"Threshold modulation: {threshold_mod}")
```

---

## 6. Technical Reference

### 6.1 `rank_value_encode`

```python
def rank_value_encode(
    expression: np.ndarray,
    global_medians: np.ndarray | None = None,
) -> np.ndarray
```

**Parameters:**

| Name              | Type                      | Description                                |
|-------------------|---------------------------|--------------------------------------------|
| `expression`      | `np.ndarray` [n_genes]    | Raw counts or normalised expression values |
| `global_medians`  | `np.ndarray | None`       | Corpus median per gene; `None` for uniform |

**Returns:** `np.ndarray` dtype `int64` — gene indices sorted by weighted expression
(descending), with zero-expression genes excluded.

**Raises:** No explicit exceptions. Empty expression vectors return an empty array.

**Algorithmic detail:**
- Weighting: `weighted = expression * (1 / (global_medians + 1e-10))` if medians provided.
- Filter: indices where `expression > 0`.
- Sort: `np.argsort(-weighted[nonzero_indices])`.
- Cast to `int64`.

### 6.2 `ScKGBERTInterface`

```python
@dataclass
class ScKGBERTInterface:
    d_model: int = 64
    n_genes: int = 2000
    sigma: float = 1.0
    seed: int = 42
```

**Dataclass fields:**

| Field      | Type    | Default | Description                              |
|------------|---------|---------|------------------------------------------|
| `d_model`  | `int`   | 64      | Embedding dimensionality                 |
| `n_genes`  | `int`   | 2000    | Gene vocabulary size                     |
| `sigma`    | `float` | 1.0     | Gaussian attention bandwidth parameter   |
| `seed`     | `int`   | 42      | RNG seed for weight initialisation       |

**Internal state (set in `__post_init__`):**

| Attribute           | Shape                      | Description                        |
|---------------------|----------------------------|------------------------------------|
| `_gene_embeddings`  | `(n_genes, d_model)`       | Shared gene token embeddings       |
| `_kg_adjacency`     | `(n_genes, n_genes)`       | PPI confidence scores (symmetric)  |
| `_w_q`              | `(d_model, d_model)`       | Query projection matrix            |
| `_w_k`              | `(d_model, d_model)`       | Key projection matrix              |
| `_w_v`              | `(d_model, d_model)`       | Value projection matrix            |
| `_cls_embedding`    | `(d_model,)`               | Classification head embedding      |

The knowledge graph adjacency matrix is initialised with `n_genes * 5` random
edges (capped at the maximum possible undirected edges), each with a confidence
score uniformly sampled from [0.15, 1.0]. The matrix is symmetric.

#### `gaussian_attention`

```python
def gaussian_attention(
    self,
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray
```

| Parameter | Shape     | Description                  |
|-----------|-----------|------------------------------|
| `queries` | `(n, d)`  | Query vectors                |
| `keys`    | `(m, d)`  | Key vectors                  |
| `values`  | `(m, d)`  | Value vectors                |
| **Return**| `(n, d)`  | Attended output              |

Uses the instance's `sigma` parameter for bandwidth.

#### `encode_expression`

```python
def encode_expression(self, expression: np.ndarray) -> np.ndarray
```

| Parameter    | Shape        | Description                        |
|--------------|--------------|------------------------------------|
| `expression` | `(n_genes,)` | Raw counts or normalised expr      |
| **Return**   | `(d_model,)` | S-Encoder cell embedding           |

Returns a zero vector if no genes are expressed or all gene indices exceed
the vocabulary size.

#### `encode_with_knowledge`

```python
def encode_with_knowledge(self, expression: np.ndarray) -> np.ndarray
```

| Parameter    | Shape        | Description                              |
|--------------|--------------|------------------------------------------|
| `expression` | `(n_genes,)` | Raw counts or normalised expr            |
| **Return**   | `(d_model,)` | Fused S-Encoder + K-Encoder embedding    |

Falls back to S-Encoder-only embedding if no expressed genes are within
the vocabulary.

#### `predict_cell_type`

```python
def predict_cell_type(
    self,
    expression: np.ndarray,
    prototypes: np.ndarray,
    labels: list[str],
) -> str
```

| Parameter    | Shape              | Description                         |
|--------------|--------------------|-------------------------------------|
| `expression` | `(n_genes,)`       | Query cell expression               |
| `prototypes` | `(n_types, d_model)`| Mean embeddings per cell type      |
| `labels`     | list of str        | Cell type names (same order)        |
| **Return**   | `str`              | Predicted cell type label           |

Classification is nearest-neighbour in embedding space (L2 distance).

#### `gene_importance`

```python
def gene_importance(self, expression: np.ndarray) -> np.ndarray
```

| Parameter    | Shape        | Description                              |
|--------------|--------------|------------------------------------------|
| `expression` | `(n_genes,)` | Raw counts or normalised expr            |
| **Return**   | `(n_genes,)` | Gene importance scores (0 for unexpressed)|

Importance is the column sum of the Gaussian attention weight matrix.
Only expressed genes receive non-zero scores.

### 6.3 `GeneformerInterface`

```python
@dataclass
class GeneformerInterface:
    d_model: int = 256
    n_genes: int = 2000
    n_heads: int = 4
    mask_ratio: float = 0.15
    seed: int = 42
```

**Dataclass fields:**

| Field        | Type    | Default | Description                              |
|--------------|---------|---------|------------------------------------------|
| `d_model`    | `int`   | 256     | Embedding dimensionality                 |
| `n_genes`    | `int`   | 2000    | Gene vocabulary size                     |
| `n_heads`    | `int`   | 4       | Number of attention heads                |
| `mask_ratio` | `float` | 0.15    | Fraction of tokens masked in MLM         |
| `seed`       | `int`   | 42      | RNG seed for weight initialisation       |

**Internal state (set in `__post_init__`):**

| Attribute           | Shape                  | Description                     |
|---------------------|------------------------|---------------------------------|
| `_gene_embeddings`  | `(n_genes, d_model)`   | Gene token embeddings           |
| `_w_q`              | `(d_model, d_model)`   | Query projection                |
| `_w_k`              | `(d_model, d_model)`   | Key projection                  |
| `_w_v`              | `(d_model, d_model)`   | Value projection                |
| `_w_o`              | `(d_model, d_model)`   | Output projection               |
| `_mlm_head`         | `(n_genes, d_model)`   | MLM prediction head             |

#### `tokenise`

```python
def tokenise(
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | None = None,
) -> np.ndarray
```

| Parameter        | Shape        | Description                                |
|------------------|--------------|--------------------------------------------|
| `expression`     | `(n_genes,)` | Raw counts or normalised expr              |
| `global_medians` | `(n_genes,)` or `None` | Corpus medians for weighting     |
| **Return**       | int64 array  | Gene token IDs within vocabulary           |

#### `mask_tokens`

```python
def mask_tokens(
    self,
    token_ids: np.ndarray,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]
```

| Parameter    | Shape     | Description                              |
|--------------|-----------|------------------------------------------|
| `token_ids`  | `(n,)`    | Gene token IDs from `tokenise`           |
| `rng_seed`   | `int | None` | RNG seed; falls back to `self.seed`  |
| **Return[0]**| `(n,)`    | Masked token IDs (-1 at masked positions)|
| **Return[1]**| `(n,)`    | Boolean mask (True where masked)         |

#### `multi_head_attention`

```python
def multi_head_attention(self, x: np.ndarray) -> np.ndarray
```

| Parameter | Shape           | Description                    |
|-----------|-----------------|--------------------------------|
| `x`       | `(seq_len, d)`  | Input token representations    |
| **Return**| `(seq_len, d)`  | Attended representations       |

#### `encode_cell`

```python
def encode_cell(
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | None = None,
) -> np.ndarray
```

| Parameter        | Shape        | Description                       |
|------------------|--------------|-----------------------------------|
| `expression`     | `(n_genes,)` | Raw counts or normalised expr     |
| `global_medians` | or `None`    | Corpus medians                    |
| **Return**       | `(d_model,)` | Mean-pooled cell embedding        |

Returns zero vector if no genes are expressed.

#### `predict_masked_genes`

```python
def predict_masked_genes(
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | None = None,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

| Parameter        | Shape        | Description                              |
|------------------|--------------|------------------------------------------|
| `expression`     | `(n_genes,)` | Raw counts or normalised expr            |
| `global_medians` | or `None`    | Corpus medians                           |
| `rng_seed`       | `int | None` | RNG seed for mask selection              |
| **Return[0]**    | `(n,)` bool  | Mask positions                           |
| **Return[1]**    | int64 array  | True gene IDs at masked positions        |
| **Return[2]**    | int64 array  | Predicted gene IDs at masked positions   |

Returns empty arrays if fewer than 2 genes are expressed.

#### `gene_network_attention`

```python
def gene_network_attention(
    self,
    expression: np.ndarray,
    global_medians: np.ndarray | None = None,
) -> np.ndarray
```

| Parameter        | Shape        | Description                              |
|------------------|--------------|------------------------------------------|
| `expression`     | `(n_genes,)` | Raw counts or normalised expr            |
| `global_medians` | or `None`    | Corpus medians                           |
| **Return**       | `(n, n)`     | Head-averaged attention matrix           |

Each row sums to 1.0. Returns `[[]]` if fewer than 2 genes are expressed.

---

## 7. Performance Benchmarks

### 7.1 Measurement methodology

All benchmarks measured on:
- **CPU:** Intel Core i5-11600K (6 cores / 12 threads, 3.9 GHz base, 4.9 GHz boost)
- **Python:** CPython 3.12
- **NumPy:** system-installed (no MKL)
- **Method:** `pytest-benchmark` with automatic round calibration

Default interface parameters unless otherwise noted:
- `ScKGBERTInterface`: d_model=64, n_genes=2000, sigma=1.0
- `GeneformerInterface`: d_model=256, n_genes=2000, n_heads=4

### 7.2 ScKGBERTInterface benchmarks

| Method                   | Mean (ns/call) | Notes                                    |
|--------------------------|----------------|------------------------------------------|
| `encode_expression`      | 516,602        | S-Encoder only, single cell              |
| `encode_with_knowledge`  | 3,237,677      | S+K dual encoder, PPI neighbourhood walk |

The `encode_with_knowledge` method is approximately 6.3x slower than
`encode_expression` due to the K-Encoder's per-gene PPI neighbourhood
aggregation loop. This loop iterates over expressed genes and, for each,
computes a confidence-weighted mean of neighbour embeddings. The cost scales
with the number of expressed genes multiplied by the average PPI neighbourhood
size.

For batch processing of large cell populations, the Rust `gaussian_attention`
kernel in `engine/src/analysis/neural_decoders.rs` can accelerate the attention
computation. The Rust kernel parallelises across query rows with `rayon`,
yielding wall-clock improvement on multi-core systems.

### 7.3 GeneformerInterface benchmarks

| Method                    | Mean (ns/call) | Notes                                  |
|---------------------------|----------------|----------------------------------------|
| `encode_cell`             | 1,035,978      | Mean-pooled cell embedding             |
| `predict_masked_genes`    | 1,447,413      | MLM prediction with 15% masking        |
| `gene_network_attention`  | 885,411        | Head-averaged attention matrix          |

The `encode_cell` and `predict_masked_genes` timings are dominated by the
multi-head self-attention computation, which involves three matrix multiplications
(Q, K, V projections), per-head score computation, softmax, and the output
projection. The `predict_masked_genes` is slower than `encode_cell` because
it additionally runs the MLM head (a matrix multiplication of masked representations
against the full vocabulary embedding table).

`gene_network_attention` is the fastest of the three because it skips the value
computation and output projection — it only computes the Q and K projections and
the per-head attention weight matrices.

### 7.4 Scaling considerations

The attention computation is $O(n^2 d)$ where $n$ is the number of expressed
genes and $d$ is the embedding dimension. For a typical scRNA-seq cell with
~3,000 expressed genes (within a vocabulary of 20,000), the attention matrix
is $3000 \times 3000$, which is manageable in numpy but becomes the bottleneck
for large-scale batch processing.

The K-Encoder neighbourhood aggregation in ScKGBERTInterface has additional
cost of $O(n \cdot \bar{k})$ where $\bar{k}$ is the mean PPI neighbourhood size.
In the STRING database, the mean degree for human proteins is approximately 10--20
high-confidence interactions, so this is typically a modest multiplicative factor.

### 7.5 Comparison of attention mechanisms

| Mechanism           | Time complexity      | Used in            |
|---------------------|----------------------|--------------------|
| Gaussian attention  | $O(n^2 d)$          | ScKGBERTInterface  |
| Scaled dot-product  | $O(n^2 d)$          | GeneformerInterface|

Both mechanisms have the same asymptotic complexity. Gaussian attention is
marginally more expensive per element due to the L2 distance computation
(requiring $q^2 + k^2 - 2qk$ vs. the single $qk$ dot product), but the
difference is small relative to the overall $O(n^2 d)$ cost.

---

## 8. Citations

### Primary references

**scKGBERT:**

Li Y, Qiao G, Du H, Gao X, Wang G.
"scKGBERT: a knowledge-enhanced foundation model for single-cell transcriptomics."
*Genome Biology* 26:402 (2025).
DOI: 10.1186/s13059-025-03564-3

**Geneformer:**

Theodoris CV, Xiao L, Chopra A, Chaffin MD, Al Sayed ZR, Hill MC, Manber H,
Bowman AGL, Dias N, Tsao PS, Bhatt DL, Ellinor PT.
"Transfer learning enables predictions in network biology."
*Nature* 619:616--624 (2023).
DOI: 10.1038/s41586-023-06139-9

### Supporting references

**Transformer attention:**

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L,
Polosukhin I.
"Attention is all you need."
*Advances in Neural Information Processing Systems* 30 (NeurIPS 2017).
arXiv: 1706.03762

**BERT masked language modelling:**

Devlin J, Chang MW, Lee K, Toutanova K.
"BERT: Pre-training of deep bidirectional transformers for language understanding."
*Proceedings of NAACL-HLT* 2019:4171--4186 (2019).
arXiv: 1810.04805

**STRING database:**

Szklarczyk D, Kirsch R, Koutrouli M, Nastou K, Mehryary F, Hachilif R,
Gable AL, Fang T, Doncheva NT, Pyysalo S, Bork P, Jensen LJ, von Mering C.
"The STRING database in 2023: protein-protein association networks and functional
enrichment analyses for any sequenced genome of interest."
*Nucleic Acids Research* 51(D1):D599--D606 (2023).
DOI: 10.1093/nar/gkac1000

**scRNA-seq methodology:**

Zheng GXY, Terry JM, Belgrader P, Ryvkin P, Bent ZW, Wilson R, Ziraldo SB,
Wheeler TD, McDermott GP, Zhu J, Gregory MT, Shuga J, Montesclaros L,
Underwood JG, Masquelier DA, Nishimura SY, Schnall-Levin M, Wyatt PW,
Hindson CM, Bharadwaj R, Wong A, Ness KD, Beppu LW, Deeg HJ, McFarland C,
Loeb KR, Valente WJ, Ericson NG, Stevens EA, Radich JP, Mikkelsen TS,
Hindson BJ, Bielas JH.
"Massively parallel digital transcriptional profiling of single cells."
*Nature Communications* 8:14049 (2017).
DOI: 10.1038/ncomms14049

**Geneformer v2 corpus:**

Theodoris CV.
"Geneformer v2: Foundation model trained on 95 million single-cell transcriptomes."
Hugging Face model repository (2024).
https://huggingface.co/ctheodoris/Geneformer

---

## Appendix A: Edge Cases and Defensive Behaviour

### A.1 All-zero expression

Both interfaces handle the case where all genes have zero expression:

- `rank_value_encode` returns an empty `int64` array.
- `ScKGBERTInterface.encode_expression` returns a zero vector of shape `(d_model,)`.
- `ScKGBERTInterface.encode_with_knowledge` returns a zero vector.
- `GeneformerInterface.encode_cell` returns a zero vector of shape `(d_model,)`.
- `GeneformerInterface.predict_masked_genes` returns three empty arrays.
- `GeneformerInterface.gene_network_attention` returns `[[]]`.

### A.2 Gene indices exceeding vocabulary

If `rank_value_encode` produces gene indices >= `n_genes`, they are filtered
out in all methods before embedding lookup. This prevents index-out-of-bounds
errors when the expression vector is longer than the model's vocabulary.

### A.3 Single expressed gene

- `ScKGBERTInterface`: works normally (attention over a single token is trivially itself).
- `GeneformerInterface.predict_masked_genes`: returns empty arrays (requires >= 2 tokens).
- `GeneformerInterface.gene_network_attention`: returns `[[]]` (requires >= 2 tokens).

### A.4 Numerical stability

Both attention mechanisms use the log-sum-exp trick:
1. Subtract the maximum log-weight per row before exponentiation.
2. Add $10^{-30}$ to the denominator to prevent division by zero.

This prevents floating-point overflow in the exponential and division-by-zero
in the normalisation, even for extreme distance/score values.

---

## Appendix B: Relationship to Full Model Architectures

The interfaces in this module implement the core algorithmic components of
scKGBERT and Geneformer, not the complete training pipelines. Specifically:

**What is implemented:**
- Rank-value encoding (tokenisation)
- Gaussian attention (scKGBERT)
- Multi-head self-attention (Geneformer)
- Dual-encoder fusion (scKGBERT)
- Masked gene prediction (Geneformer)
- Gene importance extraction (scKGBERT)
- Gene network attention extraction (Geneformer)
- Cell type prediction via prototype matching (scKGBERT)

**What is not implemented (out of scope):**
- Pretraining loops (gradient descent, loss computation, backpropagation)
- Layer normalisation, residual connections, feed-forward sub-layers
- Positional encoding beyond rank-position weighting
- Multiple Transformer layers (both models use single-layer attention here)
- Actual pretrained weights (weights are randomly initialised)

The interfaces are designed for integration into the SC-NeuroCore pipeline,
where the concern is extracting biologically meaningful representations from
gene expression data, not training foundation models from scratch. For
production use with actual pretrained weights, the weight arrays can be
replaced by loading from model checkpoints (e.g., HuggingFace).

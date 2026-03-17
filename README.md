# Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning

Code for the NeurIPS 2024 paper:
> **Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning**
> Abdulkadir Celikkanat, Andres R. Masegosa, Thomas D. Nielsen
> *Advances in Neural Information Processing Systems 37*, 2024

---

## Overview

This repository implements k-mer-based genome embedding methods for metagenomics binning. The codebase is designed around a modular registry pattern: embedders, clustering algorithms, datasets, and metrics are each independently swappable. All experiments from the paper can be reproduced with the four scripts in `scripts/`.

---

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0.

```bash
git clone https://github.com/abdcelikkanat/revisitingkmers.git
cd revisitingkmers
pip install -r requirements.txt
```

Docker (GPU):
```bash
docker build -t revisitingkmers .
```

---

## Project Structure

```
revisitingkmers/
│
├── scripts/                      # Experiment entry points (see Usage below)
│   ├── train.py                  # Train any embedding model
│   ├── evaluate_classification.py # KNN / Logistic / SVM probe on embeddings
│   ├── evaluate_clustering.py    # Multi-algorithm clustering benchmark
│   └── evaluate_checkm2.py       # Real-genome quality via CheckM2
│
├── embedders/                    # Embedding models (registry pattern)
│   ├── base.py                   # BaseEmbedder ABC + EmbeddingResult dataclass
│   ├── nonlinear.py              # NonLinearEmbedder — MLP with contrastive loss
│   ├── uncertaingen.py           # UncertainGenEmbedder — 2-phase probabilistic
│   ├── pcl.py                    # PCLEmbedder — vMF with MCInfoNCE loss
│   ├── kmer_profile.py           # KmerProfileEmbedder — raw k-mer frequency baseline
│   ├── llm.py                    # LLMEmbedder — DNABERT-2, HyenaDNA, NT via HuggingFace
│   └── __init__.py               # Registry: load_embedder(), get_embedding()
│
├── clustering/                   # Clustering algorithms (registry pattern)
│   ├── base.py                   # BaseClusterer ABC
│   ├── greedy_kmedoid.py         # Greedy K-Medoid (paper's original algorithm)
│   ├── kmedoid.py                # K-Medoid via sklearn_extra (PAM)
│   ├── dpgmm.py                  # Dirichlet Process GMM (heteroscedastic EM)
│   ├── gmm.py                    # Standard GMM (heteroscedastic EM)
│   └── __init__.py               # Registry: get_clusterer()
│
├── datasets/                     # PyTorch datasets for contrastive training
│   ├── base.py                   # BaseContrastiveDataset ABC
│   ├── paired_reads.py           # PairedReadsDataset — unlabeled paired reads CSV
│   ├── labeled_reads.py          # LabeledReadsDataset — labeled TSV
│   └── image_contrastive.py      # ImageContrastiveDataset — MNIST / CIFAR-10
│
├── features/                     # Feature extractors
│   ├── kmer.py                   # KmerFeatureExtractor — k-mer profiles, windowed
│   └── image.py                  # Image feature extractors (flat, CNN, pretrained)
│
├── metrics/
│   └── similarity.py             # Pairwise L1 / L2 / dot / vMF similarity
│
├── evaluation/                   # Evaluation pipeline utilities
│   ├── binning.py                # load_tsv_data(), evaluate_binning()
│   ├── eval_utils.py             # Threshold calibration, Hungarian alignment, F1 counts
│   ├── checkm2.py                # export_bins_to_fasta(), evaluate_checkm2()
│   └── image_benchmark.py        # evaluate_image_clustering()
│
├── utils/
│   ├── __init__.py               # filter_sequences() shared utility
│   ├── train.py                  # Generic contrastive training loop (train_contrastive)
│   └── progress.py               # tqdm wrappers, line-count caching
│
├── checkm2/                      # Bundled CheckM2 install (executable + DIAMOND database)
│   ├── bin/checkm2               # CheckM2 executable
│   └── database/CheckM2_database/uniref100.KO.1.dmnd  # DIAMOND database (~3 GB)
├── requirements.txt
└── Dockerfile
```

---

## Data

### Training data

Download the paired-reads training file (~2M read pairs, no species labels):

```bash
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
# Produces: data/dnabert/train/train_2m.csv
```

Format: one read pair per line — `left_read,right_read` (no header). Reads that appear nearby in the genome are treated as positives; negatives are sampled uniformly.

### Evaluation data

```bash
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```

After extraction the layout is:

```
data/dnabert/eval/
├── reference/
│   ├── clustering_0.tsv   ← threshold calibration (20 K seqs, 323 species)
│   ├── clustering_1.tsv
│   ├── binning_5.tsv      ← test set for binning evaluation
│   └── binning_6.tsv
├── marine/
│   └── ...                ← same structure
└── plant/
    └── ...
```

Each TSV has two tab-separated columns: `sequence` and `species_label`.

---

## Usage

### 1. Training

All models are trained through a single entry point. Use `--model` to select the architecture:

```bash
# NonLinear (bern loss, GPU)
python scripts/train.py --model nonlinear \
    --input data/dnabert/train/train_2m.csv --k 4 --dim 256 \
    --epoch 300 --lr 0.001 --loss bern \
    --neg_sample_per_pos 200 --batch_size 10000 --max_read_num 100000 \
    --device cuda \
    --output runs/nonlinear/model.model

# NonLinear (hinge loss variant)
python scripts/train.py --model nonlinear \
    --input data/dnabert/train/train_2m.csv --k 4 --dim 256 \
    --epoch 300 --lr 0.001 --loss hinge \
    --neg_sample_per_pos 200 --batch_size 10000 --max_read_num 100000 \
    --device cuda \
    --output runs/hinge/model.model

# UncertainGen — 2-phase probabilistic (GPU)
python scripts/train.py --model uncertaingen \
    --input data/dnabert/train/train_2m.csv --k 4 --dim 256 \
    --mean_epochs 50 --var_epochs 20 --lr 0.01 \
    --k_form adaptive --alpha 1.0 \
    --neg_sample_per_pos 200 --batch_size 100000 --max_read_num 100000 \
    --device cuda \
    --output runs/uncertaingen/model.model

# UncertainGen warm-started from a pretrained NonLinear model (skip Phase 1)
python scripts/train.py --model uncertaingen \
    --input data/dnabert/train/train_2m.csv --k 4 --dim 256 \
    --pretrained runs/nonlinear/model.model \
    --mean_epochs 0 --var_epochs 20 --lr 0.01 \
    --neg_sample_per_pos 200 --batch_size 100000 --max_read_num 100000 \
    --device cuda \
    --output runs/uncertaingen_ws/model.model

# PCL — vMF probabilistic (GPU)
python scripts/train.py --model pcl \
    --input data/dnabert/train/train_2m.csv --k 4 --dim 256 \
    --n_phases 0 --n_batches_per_half_phase 10000 \
    --lr 0.001 --neg_sample_per_pos 32 --batch_size 512 --max_read_num 100000 \
    --kappa_mode implicit --n_mc_samples 8 --loss_kappa_init 16 \
    --device cuda \
    --output runs/pcl/model.model
```

Full argument reference: `python scripts/train.py --help`

Each run saves files in the output folder:
- `model.model` — the trained model weights
- `model.loss` — per-epoch loss values (NonLinear, PCL)
- `mean.loss` / `variance.loss` — Phase 1 and Phase 2 loss values (UncertainGen)
- `config.json` — all CLI arguments, timestamp, git commit, and exact command used

#### Experiment folder layout

Each `--output` path creates a self-contained experiment folder:

```
runs/
└── nonlinear/                    ← one folder per training run
    ├── model.model
    ├── model.loss
    ├── config.json               ← full training config (reproducible)
    ├── embeddings/               ← auto-populated on first evaluation
    │   └── reference/
    │       ├── clustering_0.npy
    │       └── binning_5.npy
    └── results/
        ├── clustering/           ← evaluate_clustering.py output
        ├── classification/       ← evaluate_classification.py output
        └── checkm2/              ← evaluate_checkm2.py output
```

Evaluation scripts derive `embeddings/` and `results/` automatically from the model path — you only need to pass `--model_path`. Override with `--cache_dir` / `--output_dir` if needed.

---

### 2. Classification evaluation (KNN / Logistic / SVM)

Evaluates embedding quality by fitting simple classifiers on labeled sequence data. The script takes a labeled TSV (or separate train/test TSVs), embeds the sequences, and reports accuracy and F1 for all classifier configurations.

```bash
# Single model — results auto-land in runs/nonlinear/results/classification/
python scripts/evaluate_classification.py \
    --model_path runs/nonlinear/model.model \
    --test_data data/dnabert/eval/reference/clustering_0.tsv

# Separate train / test sets
python scripts/evaluate_classification.py \
    --model_path runs/nonlinear/model.model \
    --train_data data/dnabert/eval/reference/clustering_0.tsv \
    --test_data  data/dnabert/eval/reference/clustering_1.tsv

# Sweep a directory of models, KNN only
python scripts/evaluate_classification.py \
    --model_dir runs/ \
    --test_data data/dnabert/eval/reference/clustering_0.tsv \
    --classifiers knn --knn_k 1,3,5,10
```

**Key options**

| Flag | Default | Description |
|------|---------|-------------|
| `--classifiers` | `knn,logistic,svm` | Comma-separated subset to run |
| `--knn_k` | `1,3,5,10` | K values for KNN |
| `--logistic_c` | `0.01,0.1,1.0,10.0` | Regularisation sweep |
| `--svm_kernel` | `rbf` | `rbf` or `linear` |
| `--svm_c` | `0.1,1.0,10.0` | C sweep |
| `--cache_dir` | — | Cache embeddings to disk |
| `--fig_format` | `png` | `png` or `pdf` |

**Outputs** (default: `runs/<model>/results/classification/`)
```
results/classification/
├── metrics/
│   ├── classification_results.json
│   └── classification_results.csv
└── figures/
    ├── accuracy_comparison.png
    ├── confusion_<model>_knn.png
    ├── confusion_<model>_logistic.png
    └── confusion_<model>_svm.png
```

---

### 3. Clustering evaluation

Runs multiple clustering algorithms on the labeled binning data and reports precision / recall / F1 counts at thresholds 0.1–0.9. For probabilistic models (UncertainGen, PCL), also applies cluster-then-reject at multiple coverage levels.

The threshold is calibrated automatically from `clustering_0.tsv` (70th percentile of intra-class similarities, as described in the paper) — no manual tuning needed.

```bash
# All algorithms, all species, two samples — results auto-land in runs/nonlinear/results/clustering/
python scripts/evaluate_clustering.py \
    --model_path runs/nonlinear/model.model \
    --data_dir data/dnabert/eval/ \
    --species reference,marine,plant --samples 5,6 \
    --cluster_algos greedy_kmedoid,kmedoid,kmeans,dpgmm

# UncertainGen with coverage curves (cluster-then-reject)
python scripts/evaluate_clustering.py \
    --model_path runs/uncertaingen/model.model \
    --data_dir data/dnabert/eval/ \
    --species reference --samples 5,6 \
    --cluster_algos greedy_kmedoid,kmeans,dpgmm \
    --coverage_levels 100,90,80,70,60,50,40,30,20

# Sweep an entire runs directory
python scripts/evaluate_clustering.py \
    --model_dir runs/ \
    --data_dir data/dnabert/eval/ \
    --species reference --samples 5 \
    --cluster_algos kmeans
```

**Key options**

| Flag | Default | Description |
|------|---------|-------------|
| `--cluster_algos` | all four | `greedy_kmedoid`, `kmedoid`, `kmeans`, `dpgmm` |
| `--kmeans_k_range` | `50,100,200,323,500` | K values to sweep for KMeans |
| `--coverage_levels` | `100,90,…,20` | Rejection coverage levels (%) |
| `--rejection_mode` | `discard` | `discard` (exclude unassigned from metrics) or `garbage` (penalise them) |
| `--min_bin_size` | `5` | Clusters smaller than this get label −1 |
| `--cache_dir` | — | Cache embeddings between runs |

**Outputs** (default: `runs/<model>/results/clustering/`)
```
results/clustering/
├── metrics/
│   ├── clustering_results.json
│   └── clustering_results.csv   ← one row per (model, species, sample, algo, coverage)
└── figures/
    ├── f1_counts_<model>_<species>_s<N>.png    ← bar chart at each threshold
    ├── coverage_<model>_<species>_s<N>.png     ← F1>0.5 vs coverage %
    └── kmeans_sweep_<model>_<species>_s<N>.png ← F1>0.5 vs k
```

---

### 4. CheckM2 genome quality evaluation

Clusters real assembled contigs from a metagenome assembly FASTA, exports each predicted bin as a separate FASTA file, runs `checkm2 predict` to assess completeness and contamination, and summarises HQ / MQ / LQ bin counts.

#### CheckM2 setup

CheckM2 is bundled in the `checkm2/` directory (installed as a git submodule). The repository layout is:

```
checkm2/
├── bin/
│   └── checkm2                              # CheckM2 executable
├── database/
│   └── CheckM2_database/
│       └── uniref100.KO.1.dmnd              # DIAMOND database (~3 GB)
├── checkm2/                                 # Python package source
└── setup.py
```

Install the bundled version into your Python environment:

```bash
pip install ./checkm2
```

The DIAMOND database is already included at `checkm2/database/CheckM2_database/uniref100.KO.1.dmnd`. You do not need to download it separately; pass `--checkm2_db` to point to it explicitly if needed (CheckM2 auto-detects it when the package is installed from the bundled directory).

If you prefer a system-wide install instead:

```bash
mamba install -c bioconda -c conda-forge checkm2
checkm2 database --download
```

In that case, replace `--checkm2_bin checkm2/bin/checkm2` with `--checkm2_bin checkm2` in the commands below.

#### Running the evaluation

```bash
# Single model — results auto-land in runs/nonlinear/results/checkm2/
python scripts/evaluate_checkm2.py \
    --model_path runs/nonlinear/model.model \
    --fasta_path data/Fecal/eukfilt_assembly.fasta \
    --ref_data_dir data/dnabert/eval/ --ref_species reference \
    --cluster_algos greedy_kmedoid,kmeans,dpgmm \
    --checkm2_bin checkm2/bin/checkm2 \
    --checkm2_db checkm2/database/CheckM2_database/uniref100.KO.1.dmnd \
    --threads 8

# UncertainGen with 100% and 75% coverage
python scripts/evaluate_checkm2.py \
    --model_path runs/uncertaingen/model.model \
    --fasta_path data/Fecal/eukfilt_assembly.fasta \
    --ref_data_dir data/dnabert/eval/ \
    --coverage_levels 100,75 \
    --checkm2_bin checkm2/bin/checkm2 \
    --checkm2_db checkm2/database/CheckM2_database/uniref100.KO.1.dmnd
```

**Key options**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkm2_bin` | `checkm2/bin/checkm2` | Path to the CheckM2 executable |
| `--checkm2_db` | — | Path to the DIAMOND database (auto-detected if omitted) |
| `--cluster_algos` | `greedy_kmedoid,kmeans,dpgmm` | Algorithms to run |
| `--kmeans_k` | auto | Fixed k for KMeans; auto-detected from reference cluster count if omitted |
| `--coverage_levels` | `100,75,50` | Applied only to uncertainty-aware models |
| `--max_seq_len` | `20000` | Truncation length for embedding (full sequences are used for CheckM2) |
| `--threads` | `8` | CheckM2 prediction threads |

**Outputs** (default: `runs/<model>/results/checkm2/`)
```
results/checkm2/
├── bins/
│   └── <model>_<algo>_cov<N>/
│       ├── bin_0.fasta
│       ├── bin_1.fasta
│       └── ...
├── checkm2/
│   └── <model>_<algo>_cov<N>/
│       └── quality_report.tsv
├── metrics/
│   ├── checkm2_summary.json
│   └── checkm2_summary.csv
└── figures/
    ├── quality_bars_<model>.png          ← stacked HQ/MQ/LQ bars
    ├── quality_bars_all.png              ← all models compared
    └── scatter_<model>_<algo>_cov<N>.png ← completeness vs contamination
```

## Extending the Codebase

### Add a new embedder

1. Create `embedders/my_embedder.py`, implement `BaseEmbedder` (requires `embed()`, `save()`, `load()`).
2. Register it in `embedders/__init__.py`:
   ```python
   from embedders.my_embedder import MyEmbedder
   register("myembedder")(MyEmbedder)
   ```
3. It is immediately available via `--model_type myembedder` in all evaluation scripts.

### Add a new clustering algorithm

1. Create `clustering/my_algo.py`, implement `BaseClusterer` (requires `fit_predict(EmbeddingResult) -> np.ndarray`).
2. Add to the registry in `clustering/__init__.py`:
   ```python
   from clustering.my_algo import MyAlgo
   _REGISTRY["myalgo"] = MyAlgo
   ```
3. Pass `--cluster_algos myalgo` to the evaluation scripts.

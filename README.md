# ProFlow

**Pro**tein **Flow** Matching — Generative Multi-modal Proteome Inference for Precision Oncology

ProFlow predicts proteomic expression profiles from bulk transcriptomic data or H&E histology images using conditional flow matching (CFM). By bridging the gap between RNA and protein levels, it enables protein-level biological insights when only transcriptomic or imaging data is available.

## Key Features

- **Bulk Proteome Prediction** — Infer protein expression from bulk RNA-seq data, trained with paired TCGA RNA+proteomics samples.
- **Spatial Proteome Prediction** — Predict spot-level protein expression directly from H&E-stained tissue images, enabling spatially resolved proteomics without experimental measurement.
- **Survival Analysis** — Downstream unsupervised clustering and Kaplan-Meier survival analysis to identify clinically significant protein expression patterns.
- **Agent-based Interface** — LLM-powered agent for interactive proteomics analysis via a remote skill execution server.

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Bulk Prediction Pipeline                      │
│                                                                   │
│   RNA-seq ──► scGPT Encoder ──► zA (RNA latent)                  │
│                                     │                             │
│              Cancer Type ──► One-hot ──┤                          │
│                                     │                             │
│              zA + cond ──► ConditionalFlowNet ──► z_final         │
│                              (RK4, 60 steps)                      │
│                                     │                             │
│   z_final + ESM2_emb + cond ──► PredictModel ──► Protein Expr.   │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                   Spatial Prediction Pipeline                     │
│                                                                   │
│   H&E Image ──► CONCH ViT ──► Image Latent                       │
│                                     │                             │
│   Image Latent ──► HistFlowRNA ──► RNA Latent (zA)               │
│                                     │                             │
│   zA + ESM2_emb ──► ImageToProteinModel ──► Spot-level Protein   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

- **scGPT** (single-cell GPT) serves as the frozen biological encoder for both RNA and protein inputs.
- **Conditional Flow Matching** with RK4 integration learns the transport map between RNA and protein latent spaces.
- **ESM-2** provides protein sequence embeddings used as prediction priors.
- **CONCH** vision model encodes H&E histology images for spatial prediction.

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-compatible GPU (recommended)
- Conda (recommended for environment management)

### Setup

```bash
# Clone the repository
git clone https://github.com/peipp410/ProFlow.git
cd ProFlow

# Create conda environment
conda env create -f spatial_env.yml
conda activate spatial
```

### Dependencies

Core packages: PyTorch, HuggingFace Accelerate, scGPT, ESM (Meta), CONCH, scanpy, anndata, scikit-learn, scipy, pandas, tqdm.

## Project Structure

```
ProFlow/
├── train.py                  # Bulk model training (flow + prediction)
├── predict.py                # Bulk inference (RNA-only → protein)
├── dataset.py                # BulkDataset & TestDataset with scGPT preprocessing
├── encoders.py               # PredictModel & PredictModelWithCLasses
├── losses.py                 # MMD, cosine similarity, Sinkhorn OT losses
├── embedding.py              # ESM-2 protein embedding extraction
├── utils.py                  # Correlation metrics, OT projection
├── spatial/
│   ├── predict_flow.py       # Spatial model training & inference
│   ├── dataset.py            # SpatialDataset (H&E patches + RNA/protein)
│   └── st_pretrain/          # HistFlowRNA pretraining scripts
├── agent/
│   ├── server/               # XML-RPC skill execution server
│   │   ├── server.py
│   │   └── skills/
│   │       ├── bulk/         # Deployable bulk prediction skill
│   │       └── spatial/      # Deployable spatial prediction skill
│   └── client/
│       └── run.py            # LLM agent client
└── reproducibility/          # Reproducibility resources
```

## Usage

### 1. Bulk Proteome Prediction

**Training** (requires paired RNA + protein data):

```bash
python train.py
```

Edit the data paths in the `__main__` block of `train.py`:
- `rna_path`: RNA expression data (`.parquet`, genes × samples)
- `protein_path`: Protein expression data (`.csv.gz`, proteins × samples)
- `sample_group_path`: Sample metadata with cancer type labels
- `scgpt_model_path`: scGPT model directory (`vocab.json`, `args.json`, checkpoint)
- `protein_emb_path`: ESM-2 protein embeddings (`.pt` file)

**Inference** (requires trained checkpoints + RNA-only data):

```bash
python predict.py
```

Output: `result/luad_predicted_protein_expression.csv`

### 2. Spatial Proteome Prediction

**Generate spot coordinates from H&E images:**

```bash
python agent/server/skills/spatial/scripts/get_centers_HE.py \
    --input_dir <tiff_image_dir> \
    --output_dir <output_dir> \
    --tile_size 50
```

**Predict protein expression for each spot:**

```bash
python agent/server/skills/spatial/scripts/spatial_prediction_toadata.py \
    --image_dir <image_dir> \
    --coord_dir <coord_dir> \
    --output_dir <output_dir> \
    --hist_flow_rna_ckpt <checkpoint_path> \
    --final_model_ckpt <checkpoint_path>
```

Output: One `.h5ad` file per image with spot-level protein expression.

### 3. Agent System

Start the skill execution server:

```bash
python agent/server/server.py   # Listens on port 8899
```

Run the LLM agent client:

```bash
python agent/client/run.py
```

## Data Preparation

### Input Format

- **RNA expression**: `.parquet` file, genes × samples (genes as columns, samples as rows are transposed internally).
- **Protein expression**: `.csv` or `.csv.gz`, proteins × samples.
- **Sample groups**: `.csv` with sample IDs as index and cancer type as column.
- **Protein embeddings**: A single `.pt` file containing a dict mapping gene symbols to ESM-2 embedding vectors.

### scGPT Model Requirements

The scGPT model directory must contain:
- `vocab.json` — gene vocabulary
- `args.json` — model configuration
- `best_model.pt` — pretrained checkpoint (RNA or protein pretrained)

## Pretrained Models

Pretrained checkpoints are not included in this repository due to size constraints. The following models are required:

| Model | Purpose |
|---|---|
| scGPT RNA model | RNA expression encoder |
| scGPT Protein model | Protein expression encoder |
| Flow model (`flow_model.pt`) | Learned RNA→protein transport map |
| Prediction model (`pred_model.pt`) | Latent→expression decoder |
| CONCH ViT-B-16 | H&E image encoder (spatial) |
| HistFlowRNA | Image→RNA flow model (spatial) |
| ESM-2 650M | Protein sequence embeddings |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

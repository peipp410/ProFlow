# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProFlow — **Pro**tein **Flow** Matching. A generative multi-modal proteome inference framework for precision oncology. Uses conditional flow matching (CFM) with RK4 integration to predict protein expression from either bulk RNA-seq data or H&E histology images, powered by scGPT biological encoders and ESM-2 protein embeddings.

## Architecture

### Bulk Proteome Prediction (root directory)
- `train.py` — Main training script. Builds two frozen scGPT encoders (RNA + protein), trains a `ConditionalFlowNet` to learn the transport map from RNA→protein latent space, and a `PredictModelWithCLasses` that decodes latent vectors into per-protein expression values. Uses flow loss (MSE of vector field vs. true displacement) + final alignment loss (cosine + MMD) + prediction MSE loss.
- `predict.py` — Inference-only script. Loads trained flow/pred model checkpoints, takes RNA-only data through RK4 integration, outputs predicted protein expression CSV. Also contains the training/evaluation functions (commented out duplicate).
- `dataset.py` — `BulkDataset` (paired RNA+protein for training), `TestDataset` (RNA-only for inference). Both preprocess via scGPT tokenization/binning pipeline (max 1200 tokens, expression binning to 51 bins).
- `encoders.py` — `PredictModel` and `PredictModelWithCLasses`: per-protein expression heads that take latent embeddings + protein ESM-2 embeddings + cancer type labels and produce expression values via independent MLPs for each protein.
- `losses.py` — `MMDLoss` with learnable RBF kernel, `cosine_similarity_loss`, Sinkhorn OT coupling, and soft/hard InfoNCE losses.
- `embedding.py` — ESM-2 protein embedding extraction CLI (from Meta's ESM library).
- `utils.py` — Per-protein Pearson/Spearman correlation calculation, barycentric projection for OT.

### Spatial Proteome Prediction (`spatial/`)
- `spatial/predict_flow.py` — Training/inference for spatial proteomics. Uses `HistFlowRNA` (H&E image → RNA latent via CONCH vision encoder + scGPT), then an `ImageToProteinModel` that fuses histology, RNA, protein embeddings, and pathway enrichment to predict per-spot protein expression.
- `spatial/dataset.py` — `SpatialDataset`: loads H&E image patches with spatial coordinates, paired RNA/protein data.
- `spatial/st_pretrain/` — Pre-training scripts for the HistFlowRNA module (flow matching from histology images to RNA expression latent).

### Agent System (`agent/`)
- `agent/server/server.py` — XML-RPC server (port 8899) providing remote skill execution, file read/write, and a skill library browser. The skill library enumerates directories under `skills/` and serves their `SKILL.md` documentation.
- `agent/client/run.py` — LLM agent client using volcengine/deepseek models with tool-based interaction for proteomics analysis. Connects to the RPC server.
- `agent/server/skills/bulk/` — Deployable bulk prediction skill (scripts, checkpoints, data).
- `agent/server/skills/spatial/` — Deployable spatial prediction skill (scripts, checkpoints, data).

### Data Flow (bulk training)
```
RNA expression → scGPT encoder → zA (RNA latent)
Protein expression → scGPT encoder → zB (protein latent)
zA + cancer_type_label → ConditionalFlowNet (RK4, 60 steps) → z_final
z_final + protein_ESM2_emb + cancer_type_label → PredictModelWithCLasses → per-protein expression
Loss = flow_field_mse + λ_final*(cosine_sim_loss + MMD) + λ_pred*(mse + MMD)
```

## Key Technical Details

- **scGPT integration**: All RNA/protein inputs go through scGPT's tokenization (gene name→token ID), expression binning (51 bins), padding to max_length=1200, and a frozen pre-trained TransformerModel for encoding.
- **Flow matching**: `ConditionalFlowNet` takes (z_t, t, cond_vec) → vector field. Uses RK4 integration. The target vector field is `zB - zA` (RNA→protein) or `zB - noise` (noise→protein).
- **Conditioning**: Cancer type is one-hot encoded and concatenated to the latent vector. Currently only cancer type conditioning is active (protein embedding conditioning is commented out).
- **Hardcoded paths**: Many file paths in `train.py`, `predict.py` are absolute paths specific to the author's server (`/home/peijiazheng/`, `/mnt/vdd/pjz/`). These need to be changed for any new environment.
- **Seeds**: Fixed at 42 globally. `torch.backends.cudnn.deterministic = True`.
- **Protein dimension**: 512 for latents, 5120 for ESM-2 protein embeddings (383 proteins).

## Common Commands

### Bulk training
```bash
python train.py  # uses cuda:7 by default
```

### Bulk inference (RNA-only → protein)
```bash
python predict.py  # uses cuda:2 by default
```

### Spatial inference
```bash
python spatial/predict_flow.py
```

### ESM-2 protein embedding extraction
```bash
python embedding.py <model_location> <fasta_file> <output_dir> --include mean
```

### Agent server
```bash
python agent/server/server.py  # starts RPC server on port 8899
```

### Agent client
```bash
python agent/client/run.py  # requires API keys set in environment
```

## Dependencies

- **PyTorch** + HuggingFace **Accelerate**
- **scGPT** (`scgpt`) — single-cell foundation model used as RNA/protein encoder
- **ESM** (`esm`) — Meta's ESM-2 for protein sequence embeddings
- **CONCH** (`conch`) — vision model for H&E histology images (spatial module)
- **scanpy**, **anndata** — single-cell data structures
- **scipy**, **scikit-learn** — statistics and clustering
- **lora-pytorch** — LoRA adapters
- **tqdm**, **pandas**, **numpy**

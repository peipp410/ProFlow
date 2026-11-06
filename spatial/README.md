# Cleaned Protein Prediction Training Code

This folder contains a cleaned training entry point and the local model and dataset modules it uses:

- `pred_flow.py` trains the image-to-protein shared-MLP predictor and exports a training/inference checkpoint.
- `pretrained_model.py` defines the histology/RNA flow models and pretrained-weight loader.
- `dataset.py` loads image patches, aligns protein expression with protein embeddings, and prepares scGPT inputs and cached features.

The original scripts in the parent directory are left unchanged. Run commands from this folder or adjust the script path accordingly.

## Environment

Use the repository's `spatial` Conda environment, which supplies PyTorch, torchvision, pandas, NumPy, SciPy, Pillow, AnnData, scGPT, CONCH, and Accelerate, among other packages.

```bash
conda activate spatial
```

The scGPT and CONCH imports must be available in the environment. The CONCH model configuration name must also resolve through the installed CONCH package.

## Input data

Both `--train-data` and `--val-data` are CSV files whose first column contains image paths and whose remaining columns contain protein expression values. The script assigns column names from `--protein-columns`, so the number of CSV columns must equal one plus the number of supplied protein names. Every protein name must match an entry in the protein embedding file.

Image paths are resolved relative to `--image-root`. Images are center-cropped to `--crop-size` and normalized with the CONCH image normalization constants. Validation rows are grouped by sample name derived from the image path; each validation sample is evaluated every five epochs.

The scGPT model directory is expected to contain `vocab.json`, `args.json`, and `best_model.pt`. The reference embedding CSV supplies the cell-type embeddings. The CONCH and histology-flow checkpoint paths point to their respective pretrained weights.

## Run training

All paths below are examples and should be replaced with paths available in your environment. Supply the protein names in exactly the same order as the expression columns in both CSV files.

```bash
conda run -n spatial python code_clean/pred_flow.py \
  --train-data /path/to/train_norm.csv \
  --val-data /path/to/val_norm.csv \
  --checkpoint /path/to/output/protein_predictor.pt \
  --train-cache /path/to/cache/train_hist_flow_static.pt \
  --val-cache-dir /path/to/cache/validation \
  --scgpt-model-path /path/to/scgpt_model \
  --protein-embedding-path /path/to/protein_embeddings.pt \
  --image-root /path/to/images \
  --reference-embedding /path/to/celltype_reference_embeddings.csv \
  --conch-checkpoint /path/to/conch_checkpoint.bin \
  --hist-flow-checkpoint /path/to/hist_flow_checkpoint.pt \
  --protein-columns CD3 CD8 CD20 \
  --epochs 50 \
  --batch-size 2048 \
  --val-batch-size 2048 \
  --cache-batch-size 64 \
  --num-workers 1
```

Use `--help` to see all options. Important defaults include a 256-pixel crop, 512-dimensional spot features, 60 flow integration steps, and a learning rate of `0.0015`. Use `--rebuild-static-cache` to regenerate the training and validation feature caches from the supplied pretrained weights; otherwise existing caches are checked against those weights before training.

The checkpoint includes the trained prediction model, optimizer state, model configuration, preprocessing metadata, protein names and embeddings, and the static histology-flow weights required by downstream inference code.



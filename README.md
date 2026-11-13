# ProFlow

ProFlow predicts protein expression from bulk RNA-seq or H&E histology images. The repository contains bulk model training code, spatial-model training and pretraining code, packaged prediction workflows, and reproducibility scripts.

## Workflows

- **Bulk proteome training:** `train.py` trains and evaluates a conditional-flow protein predictor with five-fold cross-validation. It encodes RNA with scGPT and uses protein sequence embeddings in the prediction head.
- **Bulk prediction and survival analysis:** the packaged bulk workflow is under `agent/skills/bulk/`. Its `scripts/` directory contains `bulk_prediction.py`, `survival_analysis.py`, and `evaluate_agent.py`.
- **Spatial model training:** `spatial/pred_flow.py` trains a spot-level protein predictor from H&E image patches and exports a checkpoint for inference.
- **Spatial prediction:** the packaged spatial workflow is under `agent/skills/spatial/`. It includes coordinate generation, inference, AnnData export, annotation, and evaluation scripts.

## Repository layout

```text
.
├── data/                         # Bulk RNA, protein, and sample metadata files
├── train.py                      # Bulk five-fold training and evaluation
├── dataset.py                    # Bulk dataset and preprocessing
├── encoders.py                   # Bulk prediction models
├── embedding.py                  # Protein embedding utilities
├── losses.py                     # Training losses
├── utils.py                      # Shared utilities and metrics
├── spatial/
│   ├── pred_flow.py               # Spatial predictor training
│   ├── norm_fig.py                # ORION H&E normalization and expression extraction
│   ├── dataset.py                 # Spatial training dataset
│   ├── pretrained_model.py        # Spatial model and pretrained-weight utilities
│   └── st_pretrain/               # Histology/RNA flow pretraining
├── agent/skills/
│   ├── bulk/                      # Bulk prediction and downstream analysis tools
│   └── spatial/                   # Spatial prediction and analysis tools
└── reproducibility/               # Benchmark and figure-generation scripts
```

## Requirements and model files

Use Python 3.10 or later with a PyTorch build compatible with your hardware. The bulk and spatial workflows also rely on project-specific scientific packages such as scGPT, pandas, NumPy, and scikit-learn; spatial workflows additionally use packages including CONCH, Accelerate, AnnData, and image-processing libraries. Install the packages and model dependencies required by the workflow you plan to run. The version of used packages are listed in 'spatial_env.yml'. You can directly create the conda environment with this file.

The dataset and model checkpoints are available in https://drive.google.com/drive/folders/1yg4rU0-Ri29JDsqcWynrX34eboeeNtfj. Checkpoints are mainly used for inference, and was integrated in agent skills. Please check 'agent/skills' for detail.

## Bulk training

Run from the repository root. The defaults in `train.py` refer to a separate `train_data/` and checkpoint layout, so pass paths that match your local files and model assets. You can download default data used for training through Google Drive, then run:

```bash
python train.py \
  --rna-path data/rna_common_hvg.parquet \
  --protein-path data/protein_common.csv.gz \
  --sample-group-path data/sample_filtered_correct.csv \
  --scgpt-model-path /path/to/scgpt_model \
  --rna-checkpoint /path/to/scgpt_model/rna_state_epoch_5.pt \
  --protein-checkpoint /path/to/scgpt_model/protein_state_all.pt \
  --protein-emb-path /path/to/protein_embeddings.pt \
  --result-dir results/bulk
```

Use `python train.py --help` to see all training, preprocessing, and model options. The run performs five-fold cross-validation by default and writes fold-level predictions and metrics under the selected result directory.

### Bulk skill inference

The packaged bulk inference script is `agent/skills/bulk/scripts/bulk_prediction.py`. Prepare the RNA parquet, sample metadata CSV, protein-name pickle, protein embedding, scGPT model directory and RNA checkpoint, plus the flow and prediction checkpoints. Run from the skill directory so its helper modules resolve:

```bash
cd agent/skills/bulk
python3 scripts/bulk_prediction.py \
  --rna_path data/rna_only.parquet \
  --sample_group_path data/new_data/sample_rna_filtered.csv \
  --protein_names_path data/protein_name.pkl \
  --protein_emb_path data/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt \
  --scgpt_model_dir ckp/scgpt_model \
  --rna_scgpt_state ckp/scgpt_model/rna_state_epoch_5.pt \
  --flow_ckpt ckp/flow_model_new.pt \
  --pred_ckpt ckp/pred_model_new.pt \
  --out_csv results/bulk_protein_prediction.csv \
  --device auto
```

The output CSV is a sample-by-protein prediction matrix. Paths and checkpoint filenames should be changed to match the assets you have prepared. For downstream survival analysis, use `scripts/survival_analysis.py` with the prediction CSV and matching sample metadata; see `agent/skills/bulk/SKILL.md` for its options and example.

## Spatial data preparation, training, and prediction

Before spatial training, download the ORION tile data from [Zenodo record 15340874](https://zenodo.org/records/15340874). The archive used by `norm_fig.py` is `ORIONCRC_dataset_tile_20x.zip` (about 127 GB). For example:

```bash
mkdir -p data/zenodo
curl -L 'https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1' \
  -o data/zenodo/ORIONCRC_dataset_tile_20x.zip
7z x data/zenodo/ORIONCRC_dataset_tile_20x.zip -odata/zenodo
```

The script expects `ORIONCRC_dataset_tile_20x/train_dataframe.csv` in its current working directory, with `image_path`, `target_path`, and `nuclei_path` columns pointing to the H&E, mIF target, and nuclei-mask tiles. Arrange the extracted data at that path (or update the path in `spatial/norm_fig.py`). Supply the fitted `ckp/normalizer.pkl` separately and create the `new_he/` output directory. From the repository root, run:

```bash
mkdir -p new_he
python spatial/norm_fig.py
```

This standardizes H&E tiles into `new_he/` and writes marker expression to `ORIONCRC_dataset_tile_20x/train_dataframe_expression.csv`. The expression table contains `new_filename` and 16 marker columns (Recommend removing the PD-1 column due to poor data quality.).

The spatial training entry point is `spatial/pred_flow.py`. It requires train and validation CSVs, image and cache paths, scGPT/protein embedding assets, a reference embedding, and CONCH and histology-flow checkpoints. The script's `--help` output lists its arguments. A minimal invocation pattern is:

```bash
python spatial/pred_flow.py \
  --train-data /path/to/train.csv \
  --val-data /path/to/validation.csv \
  --checkpoint /path/to/output/spatial_predictor.pt \
  --val-cache-dir /path/to/cache/validation \
  --scgpt-model-path /path/to/scgpt_model \
  --protein-embedding-path /path/to/protein_embeddings.pt \
  --image-root /path/to/images \
  --reference-embedding /path/to/reference_embeddings.csv \
  --conch-checkpoint /path/to/conch_checkpoint.bin \
  --hist-flow-checkpoint /path/to/hist_flow_checkpoint.pt \
  --protein-columns CD3 CD8 CD20
```

### Spatial skill inference

The packaged spatial workflow runs from `agent/skills/spatial`. Prepare normalized RGB TIFF images and an exported ProFlow checkpoint; the checkpoint bundles model weights and required model metadata. Generate spot coordinates first, then run inference. For AnnData output:

```bash
cd agent/skills/spatial
python3 scripts/get_centers_HE.py \
  --input_dir data/images \
  --output_dir data/coords
python3 scripts/spatial_prediction_toadata.py \
  --image_dir data/images \
  --coord_dir data/coords \
  --output_dir results \
  --checkpoint /path/to/proflow_checkpoint.pt \
  --device auto
```

This writes one `.h5ad` result per slide. The output stores protein names as variables and spot coordinates in `obsm['spatial']`; the existing AnnData path applies the documented DBSCAN region filtering. The optional raw-GDC-SVS normalization workflow and additional output details are documented in `agent/skills/spatial/SKILL.md`. The `spatial/st_pretrain/` directory contains the dataset and training entry point for histology/RNA flow pretraining.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

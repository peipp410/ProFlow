---
name: ProFlow-Spatial
description: ProFlow spatial protein expression prediction module, inferring spot-level protein expression based on H&E images and pre-trained weights.
license: Complete terms in LICENSE.txt
---

## ProFlow-Spatial:

### 1. Project Overview

This module is used to predict the protein expression value of each spatial spot starting from H&E images, combined with pre-trained HistFlowRNA and ImageToProtein models. 
The script only executes the inference pipeline and does not contain any training steps.

### 2. Environment Configuration and Management

#### 2.1 Conda Environment Management

**Check existing environments**

```shell
# check whether the 'spatial' environment exists
conda env list | grep spatial
# activate if exists
conda activate spatial
```

**Create a new environment (if it does not exist)**

```shell
# setup environment from config file
conda env create -f spatial_env.yml
conda activate spatial
```

### 3. Core Modules and Data Preparation

**Recommended Directory Structure**

When using this project, it is recommended to prepare your directories and files according to the following structure.

```shell
spatial/
|-- SKILL.md
|-- ckp/
|   |-- HistFlowRNA_final.pt
|   |-- image_to_protein_final.pt
|   |-- reference_embedding.csv
|   |-- conch_model.bin
|   |-- scgpt_model/
|       |-- args.json
|       |-- best_model.pt
|       |-- vocab.json
|-- data/
|   |-- images/                     # *.tif
|   |-- coords/                     # valid_centers_<basename>.txt
|   |-- ESM2_embedding.pt
|-- scripts/
    |-- spatial_prediction_toadata.py
    |-- annotation.py
    |-- dataset.py
    |-- pretrained_model.py
    |-- get_centers_HE.py
```

#### 3.1 Generating Spot Spatial Coordinates (get_centers_HE.py)

The functions of this script are: 1. Find the region containing H&E tissue staining in an image based on pixel values; 2. Divide the tissue into spots of approximately 50 micrometers, and output the spatial x and y coordinates of each spot for subsequent prediction.

Before running `scripts/spatial_prediction_toadata.py`, it is recommended to use `scripts/get_centers_HE.py` to generate the available spot center coordinates for each slide from the H&E images.

Script location: `scripts/get_centers_HE.py`

**Purpose**

- Input: H&E image directory (`.tif/.tiff`)
- Output:
  - `valid_centers_<basename>.txt`: One `x,y` coordinate per line
  - `thumbnail_with_centers_<basename>.png`: Visualized thumbnail
- These `valid_centers_<basename>.txt` files can be directly used as the `--coord_dir` input for `spatial_prediction_toadata.py`.

**Parameter Details**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input_dir` | `str` | None | Input TIFF image directory |
| `--output_dir` | `str` | None | Output directory for coordinates and thumbnails |
| `--tile_size` | `int` | `50` | Grid division step size (pixels) |

**Command Example**

```bash
# If all input files are prepared according to the default parameters, this command can be run directly;
# otherwise, specify the paths to the required files using the corresponding parameters.
python3 scripts/get_centers_HE.py
```

#### 3.2 Spatial Proteome Prediction

This module provides an inference script that integrates prediction and AnnData conversion into a single pipeline. It loads trained model weights, performs protein expression prediction for each spot on every slide, applies DBSCAN-based spatial clustering for region deduplication, and outputs results as AnnData (.h5ad) files.

Script location: `scripts/spatial_prediction_toadata.py`

**Parameter Details**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image_dir` | `str` | "/home/peijiazheng/agent/server/skills/spatial/data/images" | H&E image directory (matches `--image_glob`) |
| `--coord_dir` | `str` | "/home/peijiazheng/agent/server/skills/spatial/data/coords" | Coordinate file directory |
| `--coord_prefix` | `str` | `"valid_centers_"` | Coordinate file name prefix |
| `--coord_ext` | `str` | `".txt"` | Coordinate file name extension |
| `--output_dir` | `str` | "/home/peijiazheng/agent/server/skills/spatial/results" | Output directory for prediction results |
| `--protein_emb_path` | `str` | "/home/peijiazheng/agent/server/skills/spatial/data/ESM2_embedding.pt" | Protein embedding file path (readable by `torch.load`) |
| `--scgpt_model_dir` | `str` | "/data/pjz/proteomics_back/data_model/scgpt_model" | scGPT model directory (containing `vocab.json/args.json`) |
| `--scgpt_model_state` | `str` | `None` | Optional, scGPT weight path; when empty, uses `scgpt_model_dir/best_model.pt` |
| `--conch_checkpoint_path` | `str` | "/data/pjz/proteomics_back/data_model/coach_model/pytorch_model.bin" | CONCH vision model weight path |
| `--conch_model_cfg` | `str` | `"conch_ViT-B-16"` | CONCH model configuration name |
| `--reference_embedding_path` | `str` | "/home/peijiazheng/agent/server/skills/spatial/ckp/reference_embedding.csv" | Reference cell type embedding CSV |
| `--hist_flow_rna_ckpt` | `str` | "/data/pjz/proteomics_back/data_model/tcga_coad/HistFlowRNA_final.pt" | HistFlowRNA weight path |
| `--final_model_ckpt` | `str` | "/data/pjz/proteomics_back/data_model/tcga_coad/image_to_protein_final_15.pt" | Final image-to-protein weight path |
| `--steps` | `int` | `60` | RK4 inference steps |
| `--batch_size` | `int` | `2000` | Inference batch size |
| `--patch_size` | `int` | `256` | Patch crop size |
| `--num_workers` | `int` | `1` | Number of parallel workers for DataLoader |
| `--image_glob` | `str` | `"*.tif"` | Image matching pattern |
| `--device` | `str` | `"cuda"` | Inference device |
| `--dbscan_eps` | `float` | `1500.0` | DBSCAN eps parameter for spatial region clustering |
| `--dbscan_min_samples` | `int` | `100` | DBSCAN min_samples parameter |
| `--similarity_threshold` | `float` | `0.5` | Shape similarity threshold for duplicate slice detection |

**Target Proteins Built into the Current Script**

The script currently fixes the following set of proteins for prediction in `main()`:

`PECAM1, PTPRC, CD68, CD4, FOXP3, CD8A, PTPRCRO, MS4A1, CD274, CD3E, CD163, CDH1, MKI67, KRT19, ACTA2`

**Output**

- One AnnData file per image in the output directory: `<basename>.h5ad`
- Each `.h5ad` file contains:
  - `.X`: predicted protein expression matrix (spots x proteins)
  - `.obsm['spatial']`: spatial x, y coordinates of each spot
  - `.obs`: spot-level metadata (index)
  - `.var`: protein-level metadata (protein names as feature names)
- During conversion, DBSCAN clustering is applied to filter noise spots and detect duplicate slices, retaining only the valid tissue regions

**Command Example**

```bash
# If all input files are prepared according to the default parameters, this command can be run directly;
# otherwise, specify the paths to the required files using the corresponding parameters.
python3 scripts/spatial_prediction_toadata.py
```

#### 3.3 Spatial Features Generation

This module processes all predicted `.h5ad` files from the previous step, performs dimensionality reduction, clustering, and tissue-type annotation on each slide, then extracts a comprehensive set of spatial/clinical features aggregated into a single sample-level feature matrix.

Script location: `scripts/annotation.py`

**Pipeline Summary**

For each `.h5ad` file under `results/`, the script:

1. Loads the AnnData object and runs PCA (10 components), neighbor graph construction, UMAP embedding, and Leiden clustering.
2. Performs KMeans clustering (k=10) on scaled expression values of key marker genes, then maps each cluster to one of **Tumor**, **Stroma**, **Immune**, or **Background/Unknown** based on centroid z-scores.
3. Annotates each spot with its tissue type and saves the annotated AnnData as `<basename>_annotated.h5ad`.
4. Extracts a rich feature vector via `extract_spot_features()`, including:
   - Per-protein positive fraction and mean expression
   - Global burden/infiltration/ratio scores (tumor, immune, stroma, M2 polarization, Treg/CD8, hot spot, fibrotic tumor, exhaustion, aggressive tumor)
   - Spatial distance features (tumor-immune, immune-stroma)
   - Neighborhood-based local features (tumor-immune contact, peritumoral M2/Treg/CD8/proliferation/B-cell enrichment, CD8/Treg/B-cell cluster scores, encapsulating stroma, hotspot fraction, triple interface, CD8-PD-L1 axis)

**Output**

| File | Format | Description |
| :--- | :--- | :--- |
| `data/features_df.csv` | CSV | Sample-level feature matrix (n_samples x n_features), indexed by slide basename |
| `results/<basename>_annotated.h5ad` | AnnData | Spot-level AnnData with leiden, kmeans_cluster, Tissue_Type annotations |

**Command Example**

```bash
python3 scripts/annotation.py
```

### 4. Notes

- Image files and coordinate files must be aligned by basename: `<img>.tif` corresponds to `valid_centers_<img>.txt`.
- If an image does not have a corresponding coordinate file, the script will automatically skip it.
- The `protein_emb_path` must contain embeddings corresponding to the script's built-in target proteins, otherwise an error will occur during the data loading phase.
- This script is used for inference and will not update or save training parameters.

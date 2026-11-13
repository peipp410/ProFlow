---
name: proflow-spatial
description: ProFlow spatial protein expression prediction module, inferring spot-level protein expression based on H&E images and pre-trained weights.
license: Complete terms in LICENSE.txt
---
## ProFlow-Spatial:

### 1. Project Overview

This module predicts spot-level protein expression from H&E images with the exported ProFlow TCGA checkpoint. An optional first step downloads raw H&E slides with gdc-client and normalizes them with `scripts/norm_svs.py`. It does not train or update model parameters.

### 2. Environment Configuration and Management

#### 2.1 Conda Environment Management

**Check existing environments**

```shell
# check whether the 'spatial' environment exists
conda env list | grep trident
# activate if exists
conda activate trident
```

**Create a new environment (if it does not exist)**

```shell
# setup environment from config file
conda env create -f trident_env.yml
conda activate trident
```

### 3. Core Modules and Data Preparation

**Recommended Directory Structure**

When using this project, it is recommended to prepare your directories and files according to the following structure.

```shell
spatial/
|-- SKILL.md
|-- data/
|   |-- gdc_manifest.tsv            # optional GDC H&E download manifest
|   |-- raw/                        # optional downloads: <id>/<filename>.svs
|   |-- images/                     # normalized *.tif for coordinates and inference
|   |-- coords/                     # valid_centers_<basename>.txt
|   |-- agent_eval_runs.jsonl       # annotated Agent benchmark traces
|-- ckp/normalizer.pkl              # optional fitted normalizer, supplied separately
|-- results/                        # predictions and evaluation reports
|-- scripts/
    |-- norm_svs.py                 # optional raw-SVS normalization
    |-- proflow_inference.py
    |-- spatial_prediction.py
    |-- spatial_prediction_toadata.py
    |-- evaluate_agent.py
    |-- annotation.py
    |-- dataset.py
    |-- pretrained_model.py
    |-- get_centers_HE.py
```

#### 3.1 Download and Normalize Raw H&E Slides (Optional)

**Optional first step:** use this step when starting from raw GDC H&E `.svs` slides. Skip the download if the manifest's files are already available locally; skip this entire step if suitable normalized RGB TIFFs have already been prepared. Continue with coordinate generation in Section 3.2 using those TIFFs.

Run the examples from this skill directory (`agent/skills/spatial`). Prepare:

- A GDC download manifest for H&E SVS files, exported from the GDC Data Portal. It must be a tab-separated UTF-8 file with `id` and `filename` columns; standard extra columns such as `md5`, `size`, and `state` are accepted. Use a slide-only manifest; each output basename must be unique.
- The GDC Data Transfer Tool (`gdc-client`) for downloading the raw files.
- OpenSlide's native library and `openslide-python`, NumPy, Pillow, and tifffile for normalization. Install the package used to create the fitted normalizer as well (for example, staintools and its dependencies).
- A trusted, fitted `normalizer.pkl` consistent with the normalization reference used for model training. Supply this asset separately: the script loads it and calls `transform(image)`; it does not fit a new reference. The transform must accept and return an `H x W x 3` RGB `uint8` array with the same shape. Only load trusted pickle files.

**Download raw H&E, then normalize**

```bash
mkdir -p data/raw data/images

# Download into data/raw/<id>/<filename> using the GDC manifest.
gdc-client download -m data/gdc_manifest.tsv -d data/raw

# Run after the download succeeds; provide the fitted normalizer separately.
python3 scripts/norm_svs.py \
  --manifest data/gdc_manifest.tsv \
  --input_dir data/raw \
  --output_dir data/images \
  --normalizer_path ckp/normalizer.pkl \
  --level 1
```

For controlled-access files, add `-t /path/to/gdc_token.txt` to the download command using an authorized GDC token. See the [GDC Data Transfer Tool guide](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) for installation and download options. This normalization script only reads local downloads; it does not download data itself.

**Input and output contract**

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `--manifest` | Required | GDC SVS manifest (TSV, with `id` and `filename`) |
| `--input_dir` | Skill's `data/raw` | Download root; files are read from `<input_dir>/<id>/<filename>` |
| `--output_dir` | Skill's `data/images` | Destination for TIFFs and the processing report |
| `--normalizer_path` | Required | Trusted pickle containing a fitted normalizer |
| `--level` | `1` | OpenSlide pyramid level; `0` is the original full resolution |
| `--on_block_error` | `error` | `error` fails that slide; `keep` retains failed blocks and marks the output `partial` |
| `--overwrite` | Off | Replace existing TIFFs; without it, an existing destination is reported as failed |

- Each slide produces `<output_dir>/<SVS basename>.tif`: a single-resolution RGB `uint8` TIFF with exactly the selected level's dimensions. The script normalizes four nonempty quadrants and writes their transformed pixels back in the original positions. Transparent slide regions are composited onto white. BigTIFF is used for large outputs.
- `<output_dir>/normalization_report.csv` records `id`, `filename`, `input_path`, `output_path`, `status`, `level`, `downsample`, `width`, `height`, `fallback_blocks`, and `message`. Status is `success`, `partial`, or `failed`; failed rows have no output path for this run. The report is replaced on each invocation. Per-slide failures are recorded and the remaining slides are processed.
- Exit code `0` means every slide succeeded. Any failed or partial slide returns `1`; invalid CLI arguments return `2`. Check the report before starting downstream processing. A failed slide does not replace an existing TIFF, so an older file may remain; do not treat it as a new result. `--on_block_error keep` is an explicit fallback for inspection, not a fully normalized success.
- The default level remains `1`, matching the original preprocessing script. Select a level consistent with the model's training image scale; level indices do not imply a fixed magnification across slides. Missing levels are reported as failures instead of silently changing resolution. The selected image level is loaded in memory before quadrant normalization.

After successful normalization, explicitly pass the same output image directory to coordinate generation and prediction:

```bash
python3 scripts/get_centers_HE.py \
  --input_dir data/images \
  --output_dir data/coords

python3 scripts/spatial_prediction_toadata.py \
  --image_dir data/images \
  --coord_dir data/coords \
  --output_dir results \
  --checkpoint /path/to/proflow_checkpoint.pt
```

The resulting coordinates are pixels of the normalized TIFF, not level-0 pixels of the original SVS when a downsampled level was selected. Generate coordinates from, and run prediction on, that same TIFF; do not mix original-SVS coordinates with normalized images.

#### 3.2 Generating Spot Spatial Coordinates (get_centers_HE.py)

Before running `scripts/spatial_prediction_toadata.py`, use `scripts/get_centers_HE.py` to generate spot coordinates from each H&E slide. The script uses the pretrained StarDist `2D_versatile_he` model to segment nuclei, following the same segmentation-based approach as the StarDist pipeline. It normalizes the RGB image, predicts nuclei in four overlapping quadrants, and assigns each nucleus to the quadrant containing its center to avoid duplicate detections at quadrant boundaries. Each coordinate is the midpoint of that nucleus's bounding box in full-resolution image pixel coordinates.

Script location: `scripts/get_centers_HE.py`

**Purpose**

- Input: H&E image directory (`.tif/.tiff`)
- Output:
  - `valid_centers_<basename>.txt`: One `x,y` coordinate per line, compatible with `--coord_dir` in the prediction script
  - `nucleus_spots_<basename>.csv`: StarDist label, center, bounding box, and nucleus area for each coordinate
  - `thumbnail_with_centers_<basename>.png`: H&E thumbnail with detected centers marked in red
- Coordinates refer to pixels in the original TIFF and represent detected nuclei rather than a fixed regular grid.

**Parameter Details**

| Parameter              | Type      | Default                                           | Description                                                                   |
| :--------------------- | :-------- | :------------------------------------------------ | :---------------------------------------------------------------------------- |
| `--input_dir`        | `str`   | `data/images` under the spatial skill directory | One TIFF file or a directory of TIFF images                                   |
| `--output_dir`       | `str`   | `data/coords` under the spatial skill directory | Output directory for center files, CSV details, and thumbnails                |
| `--model`            | `str`   | `2D_versatile_he`                               | Pretrained StarDist model name                                                |
| `--n_tiles`          | `str`   | None                                              | Optional StarDist tile grid (`Y,X`, e.g. `2,2`) for memory-limited slides |
| `--quadrant_overlap` | `int`   | `64`                                            | Pixel overlap around quadrant boundaries                                      |
| `--prob_thresh`      | `float` | Model default                                     | Optional StarDist probability threshold                                       |
| `--nms_thresh`       | `float` | Model default                                     | Optional StarDist non-maximum-suppression threshold                           |
| `--thumbnail_scale`  | `float` | `0.2`                                           | Thumbnail scale factor between 0 and 1                                        |

**Command Example**

```bash
# If all input files are prepared according to the default parameters, this command can be run directly;
# otherwise, specify the paths to the required files using the corresponding parameters.
python3 scripts/get_centers_HE.py
```

To process one slide or reduce StarDist memory use, pass an image file and optional tiling, for example:

```bash
python3 scripts/get_centers_HE.py \
  --input_dir data/images/sample.tif \
  --output_dir data/coords \
  --n_tiles 2,2
```

#### 3.3 Spatial Proteome Prediction

Both prediction scripts use `scripts/proflow_inference.py`, which mirrors the model construction, checkpoint loading, patch extraction, normalization, and forward pass in `inference_new/inference_tcga.py`. The exported checkpoint contains the model weights, protein names and embeddings, CONCH architecture configuration, patch size, image normalization, and flow-step count. No separate scGPT, CONCH, HistFlowRNA, or protein-embedding files are loaded.

`--device auto` (the default) selects CUDA when available and otherwise uses CPU. `--device cuda` requires a visible CUDA device. Input centers must be level-0 pixel coordinates from the paired image. The prediction matrix contains direct model outputs on the trained log1p label scale: the scripts do not apply inverse transformation, clipping, or rounding. Protein columns follow the order stored in the checkpoint.

There are two output modes:

- `scripts/spatial_prediction.py` writes a CSV with `x`, `y`, and one column per predicted protein for each slide.
- `scripts/spatial_prediction_toadata.py` writes an AnnData `.h5ad` file per slide. It retains the existing DBSCAN noise and duplicate-slice filtering before conversion; use the CSV script when every input center must be retained.

Default checkpoint: `/data/pjz/proflow/ckp/pred_flow_kl_mlp_new_checkpoint.pt`

**Parameter Details**

| Parameter                  | Type      | Default                                                      | Description                                                                 |
| :------------------------- | :-------- | :----------------------------------------------------------- | :-------------------------------------------------------------------------- |
| `--image_dir`            | `str`   | "data/images"  | H&E image directory (matches `--image_glob`)                              |
| `--coord_dir`            | `str`   | "data/coords"  | Coordinate file directory                                                   |
| `--coord_prefix`         | `str`   | `"valid_centers_"`                                         | Coordinate file name prefix                                                 |
| `--coord_ext`            | `str`   | `".txt"`                                                   | Coordinate file name extension                                              |
| `--output_dir`           | `str`   | "results"      | Output directory for prediction results                                     |
| `--checkpoint`           | `str`   | `/data/pjz/proflow/ckp/pred_flow_kl_mlp_new_checkpoint.pt` | Exported model checkpoint; includes architecture and preprocessing metadata |
| `--batch_size`           | `int`   | `2560`                                                     | Number of spot patches per inference batch                                  |
| `--image_glob`           | `str`   | `"*.tif"`                                                  | Image matching pattern                                                      |
| `--device`               | `str`   | `"auto"`                                                   | `auto`, `cuda`, or `cpu`                                              |
| `--dbscan_eps`           | `float` | `1500.0`                                                   | DBSCAN eps parameter for spatial region clustering                          |
| `--dbscan_min_samples`   | `int`   | `100`                                                      | DBSCAN min_samples parameter                                                |
| `--similarity_threshold` | `float` | `0.5`                                                      | Shape similarity threshold for duplicate slice detection                    |

**Output**

  - `.obsm['spatial']`: spatial x, y coordinates of each spot
  - `.obs`: spot-level metadata (index)
  - `.var`: protein-level metadata (protein names as feature names)
- AnnData conversion applies DBSCAN to remove noise and, when two major regions have sufficiently similar spatial fingerprints, retains only the larger region. This is the pre-existing duplicate-slice filtering behavior and can discard centers; CSV mode preserves all predictions.

**Command Example**

```bash
# If all input files are prepared according to the default parameters, this command can be run directly;
# otherwise, specify the paths to the required files using the corresponding parameters.
python3 scripts/spatial_prediction_toadata.py
```

To write an unfiltered CSV prediction table instead:

```bash
python3 scripts/spatial_prediction.py
```

#### 3.4 Quantitative Agent Evaluation

Use `scripts/evaluate_agent.py` to score a fixed benchmark of recorded spatial Agent tasks. It reports routing accuracy, end-to-end task success rate, 95% Wilson confidence intervals, per-task-type metrics, a route confusion matrix, and failure-mode counts with individual failed records. The evaluator reads annotations; it does not run the Agent or infer labels from its output.

Create a JSONL file with one record per benchmark task. `expected_route` is the reference destination and `actual_route` is the route recorded in the Agent trace. Keep route labels consistent across records (for example, `coordinate_generation`, `spatial_prediction`, and `spatial_annotation`). Mark `success` only after checking an explicit task-level acceptance criterion: coordinate generation should produce usable coordinates aligned to the source image; prediction should produce a readable result with the expected spot/protein dimensions and finite values; annotation should produce the required annotated result. Record routing and task success separately, since an incorrectly routed run may still complete the requested task.

Every failed task should have one consistent `failure_mode` label. Suggested labels are `routing_error`, `missing_input`, `coordinate_error`, `image_read_error`, `checkpoint_error`, `model_error`, `resource_error`, `invalid_output`, `timeout`, and `other`. Optional fields such as `failure_reason` and `trace_path` can be included and will be retained in `failure_cases.jsonl`.

```jsonl
{"id":"spatial_001","task_type":"spatial_prediction","expected_route":"spatial_prediction","actual_route":"spatial_prediction","success":true,"failure_mode":null}
{"id":"spatial_002","task_type":"coordinate_generation","expected_route":"coordinate_generation","actual_route":"spatial_prediction","success":false,"failure_mode":"routing_error","failure_reason":"The Agent selected the prediction route instead of generating centers."}
```

Run the evaluator with:

```bash
python3 scripts/evaluate_agent.py \
  --input_jsonl data/agent_eval_runs.jsonl \
  --out_dir results/agent_evaluation
```

The output directory contains `summary.json`, `route_confusion.csv`, `by_task_type.csv`, `failure_modes.csv`, and `failure_cases.jsonl`. Routing accuracy is the proportion whose actual route matches the expected route; task success rate is the proportion meeting the checked acceptance criterion; failure-mode fractions use failed tasks as the denominator. Use a representative held-out set, stable route/failure labels, and independently verified acceptance criteria. Report the benchmark size and confidence intervals; the evaluator does not supply benchmark annotations or establish performance by itself.

#### 3.5 Spatial Features Generation

This module processes all predicted `.h5ad` files from the previous step, performs dimensionality reduction, clustering, and tissue-type annotation on each slide, then extracts a comprehensive set of spatial/clinical features aggregated into a single sample-level feature matrix.

Script location: `scripts/annotation.py`

**Pipeline Summary**

For each `.h5ad` file under `results/`, the script:

3. Annotates each spot with its tissue type and saves the annotated AnnData as `<basename>_annotated.h5ad`.
4. Extracts a rich feature vector via `extract_spot_features()`, including:
   - Per-protein positive fraction and mean expression
   - Global burden/infiltration/ratio scores (tumor, immune, stroma, M2 polarization, Treg/CD8, hot spot, fibrotic tumor, exhaustion, aggressive tumor)
   - Spatial distance features (tumor-immune, immune-stroma)
   - Neighborhood-based local features (tumor-immune contact, peritumoral M2/Treg/CD8/proliferation/B-cell enrichment, CD8/Treg/B-cell cluster scores, encapsulating stroma, hotspot fraction, triple interface, CD8-PD-L1 axis)

**Output**

| File                                  | Format  | Description                                                                     |
| :------------------------------------ | :------ | :------------------------------------------------------------------------------ |
| `data/features_df.csv`              | CSV     | Sample-level feature matrix (n_samples x n_features), indexed by slide basename |
| `results/<basename>_annotated.h5ad` | AnnData | Spot-level AnnData with leiden, kmeans_cluster, Tissue_Type annotations         |

**Command Example**

```bash
python3 scripts/annotation.py
```

### 4. Notes

- The checkpoint determines the protein names and their order; use the same checkpoint for consistent columns across slides.
- Coordinates are expected as `x,y` values in level-0 pixels and must correspond to the paired image; centers outside the image are rejected.
- This script is used for inference and will not update or save training parameters.

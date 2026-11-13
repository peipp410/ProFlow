---
name: ProFlow_bulk
description: ProFlow_bulk is a tool that uses flow-model algorithms to predict proteomic expression from bulk transcriptomic data.
license: Complete terms in LICENSE.txt
---
## ProFlow:

### 1. Project Overview

This project introduces ProFlow (bulk), a computational tool designed to predict proteomic expression profiles from bulk transcriptomic data using a simulation-free conditional flow-matching model. The predictor starts from the RNA scGPT embedding, integrates the learned velocity field with RK4, and decodes protein expression with an ESM2-conditioned residual prediction head. It also provides downstream analysis pipelines, including unsupervised clustering and survival analysis.

### 2. Environment Configuration and Management

#### 2.1 Conda Environment Management

**Check existing environments**

```shell
# check whether the 'scgpt' environment exists
conda env list | grep scgpt
# activate if exists
conda activate scgpt
```

#### 2.2 GPU availability check

Run the check inside the same activated environment that will run inference. The
PyTorch check is authoritative for this script; `nvidia-smi` is only an optional
driver diagnostic and may be unavailable even when PyTorch can access CUDA.

```shell
conda activate scgpt
python -c "import torch; print('torch:', torch.__version__); print('cuda_build:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use `--device auto` for inference. It selects CUDA when
`torch.cuda.is_available()` is true and otherwise falls back to CPU. Use
`--device cuda` only when a CUDA device is required; it fails early if no GPU is
visible. Use `--device cpu` to force CPU execution.

**Create a new environment (if it does not exist)**

```shell
# setup environment from config file
conda env create -f scgpt_env.yml
```

### 3. Core Modules and Corresponding Data Preparation

This model can predict bulk proteomic profiles from bulk transcriptomic data, and can also predict spatial protein expression for each spot from H&E images. The methods for these two modes and the required files are introduced below.

#### 3.1 Bulk Proteome Prediction

This module provides an inference script that can be directly called to load trained model weights and prepared input data, outputting the predicted protein expression matrix:

Script location: scripts/bulk_prediction.py

**Recommended directory structure**

```shell
bulk/
├── SKILL.md
├── ckp
│   ├── flow_model_new.pt
│   ├── pred_model_new.pt
│   └── scgpt_model
│       ├── args.json
│       ├── best_model.pt
│       └── vocab.json
├── data
│   ├── ESM2_embedding.pt
│   ├── protein_name.pkl
│   ├── rna_only_hvg.parquet
│   └── sample_rna.csv
│   └── agent_eval_runs.jsonl  # create this annotated benchmark file for evaluation
├── results
└── scripts
    ├── bulk_prediction.py
    ├── evaluate_agent.py
    ├── dataset.py
    ├── encoders.py
    ├── losses.py
    └── utils.py
```

**Parameter details**

| Parameter                | Type    | Default                               | Description                                                                                                          |
| :----------------------- | :------ | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------- |
| `--rna_path`           | `str` | "data/rna\_only.parquet"              | Input path for the RNA expression data file                                                                          |
| `--sample_group_path`  | `str` | "data/sample\_rna.csv"                | Path to the sample grouping or metadata file                                                                         |
| `--protein_names_path` | `str` | "data/protein_name.pkl"               | Path to the protein name list file                                                                                   |
| `--protein_emb_path`   | `str` | "data/ESM2_embedding.pt"              | Path to the protein feature/embedding file                                                                           |
| `--scgpt_model_dir`    | `str` | "ckp/scgpt_model/"                    | Directory path containing scGPT model files                                                                          |
| `--rna_scgpt_state`    | `str` | "ckp/scgpt_model/best_model.pt"       | Path to the state dictionary/weight file of the RNA scGPT model                                                      |
| `--flow_ckpt`          | `str` | "ckp/flow_model_new.pt"               | Path to the simulation-free flow velocity checkpoint                                                                 |
| `--pred_ckpt`          | `str` | "ckp/pred_model_new.pt"               | Path to the shared residual prediction-head checkpoint                                                               |
| `--steps`              | `int` | `60`                                | Number of iteration steps for model inference or generation                                                          |
| `--batch_size`         | `int` | `512`                               | Batch size                                                                                                           |
| `--device`             | `str` | `"auto"`                            | Running device: automatically select CUDA when available, otherwise CPU; choices are `auto`, `cuda`, and `cpu` |
| `--zero_based_groups`  | flag    | off                                   | Treat sample groups as zero-based; training metadata is one-based by default                                         |
| `--out_csv`            | `str` | "results/bulk_protein_prediction.csv" | Output path for the final prediction result CSV file                                                                 |

**Output**

- `--out_csv`: Predicted protein expression matrix (samples × proteins, with column names as protein_names)

**Command example**

```bash
# If all input files and checkpoints are prepared according to the default parameters, this command can be run directly;
# otherwise, specify the paths for the required parameters.
python3 scripts/bulk_prediction.py --device auto
```

#### 3.2 Quantitative Agent Evaluation

Use `scripts/evaluate_agent.py` to score a fixed, annotated benchmark of Agent runs. It reports routing accuracy, end-to-end task success rate, 95% Wilson confidence intervals, metrics by task type, a route confusion matrix, and failure-mode counts plus the individual failed records. The evaluator consumes recorded runs; it does not execute or route tasks itself.

Create a JSONL file with one record per benchmark task. `expected_route` is the reference label and `actual_route` is the route recorded by the Agent. `success` must reflect an explicitly checked task-level acceptance criterion (for example, the required output exists and passes validation), rather than whether the Agent merely returned a response. Every failed task should have a `failure_mode` label; use a consistent taxonomy such as `routing_error`, `missing_input`, `data_error`, `model_error`, `invalid_output`, `timeout`, and `other`.

```jsonl
{"id":"bulk_001","task_type":"bulk_prediction","expected_route":"bulk_prediction","actual_route":"bulk_prediction","success":true,"failure_mode":null}
{"id":"survival_001","task_type":"survival_analysis","expected_route":"survival_analysis","actual_route":"bulk_prediction","success":false,"failure_mode":"routing_error"}
```

Run the evaluator with:

```bash
python3 scripts/evaluate_agent.py \
  --input_jsonl data/agent_eval_runs.jsonl \
  --out_dir results/agent_evaluation
```

The output directory contains `summary.json` (overall rates, confidence intervals, route confusion matrix, and per-task-type metrics), `route_confusion.csv`, `by_task_type.csv`, `failure_modes.csv`, and `failure_cases.jsonl`. Routing accuracy is the fraction of tasks where the actual route matches the expected route. Task success rate is the fraction meeting the benchmark acceptance criterion. Failure-mode fractions are calculated among failed tasks only. Use a representative held-out task set, fixed route and failure labels, and independently verified success criteria when reporting these results; the score is only as reliable as the benchmark annotations.

#### 3.3 Survival Analysis After Proteome Prediction

After obtaining the predicted bulk protein expression matrix, further unsupervised clustering can be performed. Combined with TCGA survival data, this evaluates prognostic differences among different clusters to identify clinically meaningful protein expression patterns.

**Script logic description**:

1) Data preparation

- `--exp_path`: use the prediction CSV generated for the current request. If the bulk prediction step used a custom `--out_csv`, pass that exact path. If no prediction was generated in this request and no file was specified, use `results/bulk_protein_prediction.csv` when it exists; otherwise select the newest completed prediction CSV in `results/` and report the selected path.
- `--sample_path`: use the sample metadata CSV containing the selected prediction matrix's sample IDs and a `name` cancer-type column. In the standard bulk layout this is `data/sample_rna.csv`. Check that its sample IDs cover the prediction rows before running the analysis, and pass the selected path explicitly.
- Automatically download: TCGA survival data after expression-based clustering is complete.

2) Clustering and dimensionality reduction

- Select the requested cancer type from the expression matrix and cluster all available expression samples first. Do not remove samples based on survival-data availability before PCA or Leiden clustering.
- PCA (50 dimensions)
- KNN graph (`n_neighbors=20`, `n_pcs=30`)
- Leiden clustering (multi-resolution search for the optimal)

3) Survival analysis

After Leiden labels are fixed, match cluster assignments to TCGA survival records by sample ID. Retain matched records with valid nonnegative `OS.time` and binary `OS` values, then execute for each cluster with survival data:

- Current cluster vs other samples
- Log-rank test → p-value
- Kaplan-Meier fitting → median survival time

Survival trend decision rules:

- Cluster median survival > others → better survival
- Cluster median survival < others → worse survival
- Equal → similar

4) Differential protein analysis

Use the Wilcoxon rank-sum test across all clustered cancer samples, including samples without survival records:

```python
scanpy.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')
```

Filtering criteria:

- adjusted p-value < 0.05
- Select the top 20 proteins for each cluster (ranked by significance)

**Output result format**:

1) Significant clusters (p < 0.05)

```text
Cluster 2: p = 0.0031, this group has worse survival
Cluster 5: p = 0.012, this group has better survival
```

2) Corresponding key proteins (Top 20)

```text
Cluster 2: TP53,EGFR,CDK1,CCNB1,...
Cluster 5: ALB,APOA1,HP,...
```

**Command example**:

```bash
python3 scripts/survival_analysis.py \
  --cancer_type UCEC \
  --sample_path data/sample_rna.csv \
  --exp_path results/bulk_protein_prediction.csv
```

**Parameter description**

| Parameter         | Type | Required | Description                                                                                                                                                                                                                                   |
| :---------------- | :--- | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--cancer_type` | str  | Yes      | TCGA cancer type (e.g., UCEC, LUAD, BRCA)                                                                                                                                                                                                     |
| `--sample_path` | str  | No       | Matching sample metadata CSV with sample IDs as the index and a `name` cancer-type column; standard default is `data/sample_rna.csv`. The agent selects the metadata file matching the chosen prediction output and passes it explicitly. |
| `--exp_path`    | str  | No       | Sample-by-protein prediction CSV. The agent passes the current task's prediction output explicitly; the script default is `results/bulk_protein_prediction.csv`.                                                                            |

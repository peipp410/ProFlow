---
name: ProFlow_bulk
description: ProFlow_bulk is a tool that uses flow-model algorithms to predict proteomic expression from bulk transcriptomic data.
license: Complete terms in LICENSE.txt
---

## ProFlow:

### 1. Project Overview

This project introduces ProFlow (bulk), a computational tool designed to predict proteomic expression profiles from bulk transcriptomic data using advanced flow-matching models. By bridging the gap between RNA and protein levels, it enables researchers to gain deeper insights into protein-level biological mechanisms when only transcriptomic data is available. Additionally, it provides downstream analysis pipelines, including unsupervised clustering and survival analysis, to identify clinically significant protein expression patterns and potential biomarkers.

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
```

#### 


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
│   ├── flow_model.pt
│   ├── pred_model.pt
│   └── scgpt_model
│       ├── args.json
│       ├── best_model.pt
│       └── vocab.json
├── data
│   ├── ESM2_embedding.pt
│   ├── protein_name.pkl
│   ├── test_all_rna.parquet
│   └── test_all_samples.csv
├── results
└── scripts
    ├── bulk_prediction.py
    ├── dataset.py
    ├── encoders.py
    ├── losses.py
    └── utils.py
```

**Parameter details**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--rna_path` | `str` | "data/test_all_rna.parquet" | Input path for the RNA expression data file |
| `--sample_group_path` | `str` | "data/test_all_samples.csv" | Path to the sample grouping or metadata file |
| `--protein_names_path` | `str` | "data/protein_name.pkl" | Path to the protein name list file |
| `--protein_emb_path` | `str` | "data/ESM2_embedding.pt" | Path to the protein feature/embedding file |
| `--scgpt_model_dir` | `str` | "ckp/scgpt_model/" | Directory path containing scGPT model files |
| `--rna_scgpt_state` | `str` | "ckp/scgpt_model/best_model.pt" | Path to the state dictionary/weight file of the RNA scGPT model |
| `--flow_ckpt` | `str` | "ckp/flow_model.pt" | Path to the flow matching model/flow model checkpoint |
| `--pred_ckpt` | `str` | "ckp/pred_model.pt" | Path to the prediction model checkpoint |
| `--num_classes` | `int` | `32` | Total number of classification categories |
| `--steps` | `int` | `60` | Number of iteration steps for model inference or generation |
| `--batch_size` | `int` | `512` | Batch size |
| `--device` | `str` | `"cuda"` | Running device, defaults to GPU ("cuda") |
| `--out_csv` | `str` | "results/bulk_protein_prediction.csv" | Output path for the final prediction result CSV file |

**Output**

- `--out_csv`: Predicted protein expression matrix (samples × proteins, with column names as protein_names)

**Command example**

```bash
# If all input files are prepared according to the default parameters, this command can be run directly;
# otherwise, you need to specify the paths for the required parameters.
python3 scripts/bulk_prediction.py
```

#### 3.2 Survival Analysis After Proteome Prediction

After obtaining the predicted bulk protein expression matrix, further unsupervised clustering can be performed. Combined with TCGA survival data, this evaluates prognostic differences among different clusters to identify clinically meaningful protein expression patterns.

**Script logic description**:

1) Data preparation

- Input: `results/bulk_protein_prediction.csv` (output from the previous module)
- Input: `data/test_all_samples.csv` (sample information)
- Automatically download: TCGA survival data

2) Clustering and dimensionality reduction

- PCA (50 dimensions)
- KNN graph (n_neighbors=30)
- Leiden clustering (multi-resolution search for the optimal)

3) Survival analysis

Execute for each cluster:

- Current cluster vs other samples
- Log-rank test → p-value
- Kaplan-Meier fitting → median survival time

Survival trend decision rules:

- Cluster median survival > others → better survival
- Cluster median survival < others → worse survival
- Equal → similar

4) Differential protein analysis

Use the Wilcoxon rank-sum test:

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
python3 survival_analysis.py --cancer_type UCEC
```

**Parameter description**

| Parameter | Type | Required | Description |
| :-------------- | :--- | :--- | :------------------------------- |
| `--cancer_type` | str  | Yes  | TCGA cancer type (e.g., UCEC, LUAD, BRCA) |
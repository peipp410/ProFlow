"""Spatial pretraining datasets, scGPT input collation and data loaders."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scgpt.tokenizer import GeneVocab
from skimage.io import imread
from torch.utils.data import ConcatDataset, DataLoader, default_collate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    """Pad, sample and bin expression inputs for scGPT."""

    do_padding: bool = True
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_binning: bool = True
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")
        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                f"`keep_first_n_tokens` must be between 0 and `max_length` ({self.max_length})."
            )

    def collate(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of examples (simplified for direct numpy array input).
        """
        device = "cpu"
        if not isinstance(examples[0], Mapping):
            raise NotImplementedError
        max_ori_len = max((len(example["genes"]) for example in examples))
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len
        padded_genes = []
        padded_expressions = []
        for example in examples:
            genes = torch.from_numpy(example["genes"]).long()
            expressions = torch.from_numpy(example["expressions"]).float()
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :], n_bins=51
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_genes.append(genes)
            padded_expressions.append(expressions)
        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)
        data_dict = {"gene": padded_genes, "expr": padded_expressions}
        return data_dict

    def _sample_or_truncate_plus_pad(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(genes) == max_length:
            return (genes, expressions)
        if len(genes) > max_length:
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            return (genes[:max_length], expressions[:max_length])
        return self._pad(genes, expressions, max_length)

    def _sample(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes))[:max_length]
            return (genes[indices], expressions[indices])
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return (genes[indices], expressions[indices])

    def _pad(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),), self.pad_token_id, dtype=genes.dtype
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                ),
            ]
        )
        return (genes, expressions)


def _digitize(x: np.ndarray, bins: np.ndarray, right: bool = False) -> np.ndarray:
    """Helper function for binning, adopted from numpy's digitize."""
    if len(x) == 0:
        return np.array([], dtype=np.int64)
    if bins.ndim != 1:
        raise ValueError("bins must be 1-dimensional.")
    if not np.all(np.diff(bins) >= 0):
        raise ValueError("bins must be monotonically increasing or decreasing.")
    return np.digitize(x, bins, right=right)


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


class SpatialDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path,
        spatial_pos_path,
        count_mtx_path,
        celltype_mask_path,
        scgpt_model_path,
        augment_factor=1,
    ):
        self.whole_image = imread(image_path)
        if self.whole_image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        self.spatial_pos_csv = pd.read_csv(
            spatial_pos_path, sep=",", header=0, index_col=0
        )
        self.barcode_tsv = self.spatial_pos_csv.index.tolist()
        self.count_matrix = pd.read_parquet(count_mtx_path, engine="pyarrow")
        self.count_matrix = self.count_matrix.T
        self.augment_factor = augment_factor
        self.celltype_mask = pd.read_parquet(celltype_mask_path, engine="pyarrow")
        self.celltype_mask = self.celltype_mask.values
        self.scgpt_model_dir = Path(scgpt_model_path)
        if len(self.spatial_pos_csv) == 0 or len(self.barcode_tsv) == 0:
            raise ValueError("CSV/TSV files are empty or improperly formatted.")
        if self.count_matrix.shape[0] != len(self.barcode_tsv):
            raise ValueError(
                f"Count matrix row count must match the number of barcodes. {count_mtx_path} got {self.count_matrix.shape[0]} but barcodes {len(self.barcode_tsv)}"
            )
        all_zeros = (self.count_matrix.abs() < 1e-09).all(axis=1)
        self.zero_rows_index = all_zeros[all_zeros].index.tolist()
        if self.zero_rows_index:
            self.count_matrix.loc[
                self.zero_rows_index, self.count_matrix.columns[:2]
            ] = 1
        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.non_augmentations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.input_gene_ids, self.expressions, self.src_key_padding_mask = (
            self._construct_scgpt_input(self.count_matrix, self.scgpt_model_dir)
        )
        self.failed_indices = []

    def _construct_scgpt_input(self, input_count_matrix, scgpt_model_dir):
        vocab_file = scgpt_model_dir / "vocab.json"
        model_config_file = scgpt_model_dir / "args.json"
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        adata = ad.AnnData(input_count_matrix)
        adata.var["gene"] = adata.var.index.tolist()
        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata.var["gene"]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        vocab.set_default_index(vocab["<pad>"])
        genes = adata.var["gene"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        count_matrix = adata.X
        count_matrix = (
            count_matrix
            if isinstance(count_matrix, np.ndarray)
            else count_matrix.toarray()
        )
        all_examples = []
        for i in range(count_matrix.shape[0]):
            row = count_matrix[i]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = gene_ids[nonzero_idx]
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs.get("pad_value", 0))
            all_examples.append({"genes": genes, "expressions": values})
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs.get("pad_token", "<pad>")],
            pad_value=model_configs.get("pad_value", 0),
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collated_data = collator.collate(all_examples)
        input_gene_ids = collated_data["gene"]
        expressions = collated_data["expr"]
        src_key_padding_mask = input_gene_ids.eq(
            vocab[model_configs.get("pad_token", "<pad>")]
        )
        input_gene_ids = torch.tensor(input_gene_ids, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float32)
        src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)
        return (input_gene_ids, expressions, src_key_padding_mask)

    def transform(self, image, augment_idx):
        image = Image.fromarray(image)
        if augment_idx > 0:
            image = self.augmentations(image)
        else:
            image = self.non_augmentations(image)
        return image

    def __getitem__(self, idx):
        actual_idx = idx % len(self.barcode_tsv)
        augment_idx = idx // len(self.barcode_tsv)
        if actual_idx in self.failed_indices:
            return None
        item = {}
        barcode = self.barcode_tsv[actual_idx]
        if barcode in self.zero_rows_index:
            logging.warning(f"Barcode {barcode} is in zero rows index. Skipping.")
            self.failed_indices.append(idx)
            return None
        spatial_info = self.spatial_pos_csv.loc[[barcode]]
        input_gene_ids = self.input_gene_ids[actual_idx]
        expressions = self.expressions[actual_idx]
        src_key_padding_mask = self.src_key_padding_mask[actual_idx]
        spot_celltype_mask = self.celltype_mask[actual_idx]
        if spatial_info.empty:
            logging.warning(
                f"Warning: No spatial information found for barcode {barcode}. Skipping."
            )
            self.failed_indices.append(idx)
            return None
        try:
            v1 = int(spatial_info["pixel_x"].iloc[0])
            v2 = int(spatial_info["pixel_y"].iloc[0])
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Error processing barcode {barcode}: {e}. Skipping.")
            self.failed_indices.append(idx)
            return None
        r_start, r_end = (v1 - 128, v1 + 128)
        c_start, c_end = (v2 - 128, v2 + 128)
        img_h, img_w = self.whole_image.shape[:2]
        if r_start < 0:
            r_start, r_end = (0, 256)
        if r_end > img_h:
            r_start, r_end = (img_h - 256, img_h)
        if c_start < 0:
            c_start, c_end = (0, 256)
        if c_end > img_w:
            c_start, c_end = (img_w - 256, img_w)
        image_patch_raw = self.whole_image[r_start:r_end, c_start:c_end]
        if image_patch_raw.shape[0] != 256 or image_patch_raw.shape[1] != 256:
            self.failed_indices.append(idx)
            return None
        image_patch = self.transform(image_patch_raw, augment_idx)
        item["image"] = image_patch
        item["input_gene_ids"] = input_gene_ids
        item["expressions"] = expressions
        item["src_key_padding_mask"] = src_key_padding_mask
        item["spot_celltype_mask"] = torch.tensor(spot_celltype_mask).float()
        item["actual_idx"] = torch.tensor(actual_idx, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.barcode_tsv) * self.augment_factor


def custom_collate(batch):
    """Exclude invalid samples; return None if the entire batch is invalid."""
    batch = [sample for sample in batch if sample is not None]
    return default_collate(batch) if batch else None


def _build_spatial_split(sample_ids, data_root, scgpt_model_path):
    """Load each tissue in a split using the shared directory layout."""
    return ConcatDataset(
        [
            SpatialDataset(
                image_path=data_root
                / "img_norm"
                / f"Visium_HD_Human_Colon_Cancer_{sample_id}_tissue_image.tif",
                spatial_pos_path=data_root / "st" / f"spot_{sample_id}.csv",
                count_mtx_path=data_root / "counts" / f"oricount_{sample_id}.parquet",
                celltype_mask_path=data_root
                / "enrich"
                / f"enrich_{sample_id}_sm.parquet",
                scgpt_model_path=scgpt_model_path,
            )
            for sample_id in sample_ids
        ]
    )


def build_loaders_spatial(
    batch_size=128,
    *,
    data_root="/macroverse-nas/pjz/crc_codex/pretrain/data",
    scgpt_model_path="/macroverse-nas/pjz/proteomics/data_model/scgpt_model",
):
    """Train on P0/P1/P2 and evaluate on held-out P5_T tissue."""
    data_root = Path(data_root)
    train_dataset = _build_spatial_split(
        ("P0", "P1", "P2"), data_root, scgpt_model_path
    )
    test_dataset = _build_spatial_split(("P5_T",), data_root, scgpt_model_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )
    return train_loader, test_loader

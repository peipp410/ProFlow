import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scgpt.tokenizer import GeneVocab
from torch.utils.data import Dataset
from torchvision import transforms


LOGGER = logging.getLogger(__name__)


@dataclass
class DataCollator:
    do_padding: bool = True
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1

    def __post_init__(self):
        if self.do_padding and (self.pad_token_id is None or self.max_length is None):
            raise ValueError("pad_token_id and max_length are required when padding is enabled")
        if not 0 < self.mlm_probability < 1:
            raise ValueError("mlm_probability must be between 0 and 1")
        if self.max_length is not None and not 0 <= self.keep_first_n_tokens <= self.max_length:
            raise ValueError("keep_first_n_tokens must be between 0 and max_length")

    def collate(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        if not examples or not isinstance(examples[0], Mapping):
            raise ValueError("examples must be a non-empty list of mappings")

        max_length = min(self.max_length, max(len(item["genes"]) for item in examples))
        padded_genes = []
        padded_expressions = []
        for example in examples:
            genes = torch.as_tensor(example["genes"], dtype=torch.long)
            expressions = torch.as_tensor(example["expressions"], dtype=torch.float32)
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    expressions[self.keep_first_n_tokens :], n_bins=51
                )
            genes, expressions = self._sample_or_truncate_plus_pad(genes, expressions, max_length)
            padded_genes.append(genes)
            padded_expressions.append(expressions)

        expressions = torch.stack(padded_expressions)
        if self.do_mlm:
            masked_expressions = self._mask(expressions)
        else:
            masked_expressions = expressions
        return {
            "gene": torch.stack(padded_genes),
            "expr": expressions,
            "masked_expr": masked_expressions,
        }

    def _mask(self, expressions: torch.Tensor) -> torch.Tensor:
        probability = torch.full(expressions.shape, self.mlm_probability)
        probability[expressions.eq(self.pad_value)] = 0
        probability[:, : self.keep_first_n_tokens] = 0
        mask = torch.bernoulli(probability).bool()
        return expressions.masked_fill(mask, self.mask_value)

    def _sample_or_truncate_plus_pad(self, genes, expressions, max_length):
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            return genes[:max_length], expressions[:max_length]
        return self._pad(genes, expressions, max_length)

    def _sample(self, genes, expressions, max_length):
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes))[:max_length]
        else:
            first = self.keep_first_n_tokens
            indices = torch.randperm(len(genes) - first)[: max_length - first]
            indices = torch.cat([torch.arange(first), indices + first])
        return genes[indices], expressions[indices]

    def _pad(self, genes, expressions, max_length):
        padding = max_length - len(genes)
        return (
            torch.cat([genes, torch.full((padding,), self.pad_token_id, dtype=genes.dtype)]),
            torch.cat([expressions, torch.full((padding,), self.pad_value, dtype=expressions.dtype)]),
        )


def _digitize(x: np.ndarray, bins: np.ndarray, right: bool = False) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.int64)
    if bins.ndim != 1:
        raise ValueError("bins must be one-dimensional")
    if not np.all(np.diff(bins) >= 0):
        raise ValueError("bins must be monotonically increasing")
    return np.digitize(x, bins, right=right)


def binning(row: Union[np.ndarray, torch.Tensor], n_bins: int):
    original_dtype = row.dtype
    is_numpy = isinstance(row, np.ndarray)
    values = row if is_numpy else row.detach().cpu().numpy()
    if values.size == 0:
        return row.clone() if isinstance(row, torch.Tensor) else row.copy()
    if values.max() == 0:
        LOGGER.warning("Input expression row contains only zeros")
        result = np.zeros_like(values, dtype=original_dtype)
    elif values.min() <= 0:
        nonzero = np.flatnonzero(values)
        bins = np.quantile(values[nonzero], np.linspace(0, 1, n_bins - 1))
        result = np.zeros_like(values, dtype=np.int64)
        result[nonzero] = _digitize(values[nonzero], bins)
    else:
        bins = np.quantile(values, np.linspace(0, 1, n_bins - 1))
        result = _digitize(values, bins).astype(original_dtype)
    return result if is_numpy else torch.from_numpy(result)


class ProteinExpressionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        scgpt_model_path: Union[str, Path],
        image_root: Union[str, Path],
        transform=None,
        protein_emb_path: Optional[Union[str, Path]] = None,
        static_feature_path: Optional[Union[str, Path]] = None,
        crop_size: int = 256,
    ):
        self.dataframe = dataframe
        self.image_paths = dataframe.iloc[:, 0].to_numpy()
        self.protein_expression = dataframe.iloc[:, 1:]
        self.protein_names = dataframe.columns[1:]
        self.transform = transform
        self.image_root = Path(image_root)
        self.crop_size = crop_size
        self.non_augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        self.scgpt_model_dir = Path(scgpt_model_path)
        if protein_emb_path is None:
            raise ValueError("protein_emb_path is required")

        protein_embedding = torch.load(protein_emb_path, map_location="cpu")
        self.protein_emb, self.protein_exp, self.protein_names = self._extract_embeddings(
            protein_embedding, self.protein_expression
        )
        (
            self.input_protein_ids,
            self.protein_expressions,
            self.src_key_padding_mask_protein,
        ) = self._construct_scgpt_input(self.protein_exp)
        self.protein_expression = np.log1p(self.protein_exp.to_numpy()).astype(np.float32)

        self.static_features = None
        if static_feature_path is not None:
            self.static_features = torch.load(static_feature_path, map_location="cpu")
            self._validate_static_features()

    def _construct_scgpt_input(self, input_count_matrix):
        vocab = GeneVocab.from_file(self.scgpt_model_dir / "vocab.json")
        for token in ("<pad>", "<cls>", "<eoc>"):
            if token not in vocab:
                vocab.append_token(token)
        with open(self.scgpt_model_dir / "args.json", encoding="utf-8") as handle:
            model_config = json.load(handle)
        vocab.set_default_index(vocab["<pad>"])

        adata = ad.AnnData(input_count_matrix)
        adata.var["gene"] = adata.var.index.tolist()
        adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var["gene"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        gene_ids = np.asarray(vocab(adata.var["gene"].tolist()), dtype=int)
        count_matrix = adata.X
        if not isinstance(count_matrix, np.ndarray):
            count_matrix = count_matrix.toarray()

        examples = []
        for row in count_matrix:
            nonzero = np.flatnonzero(row)
            examples.append({
                "genes": np.insert(gene_ids[nonzero], 0, vocab["<cls>"]),
                "expressions": np.insert(row[nonzero], 0, model_config.get("pad_value", 0)),
            })

        pad_token = model_config.get("pad_token", "<pad>")
        collator = DataCollator(
            pad_token_id=vocab[pad_token],
            pad_value=model_config.get("pad_value", 0),
            do_mlm=False,
            max_length=900,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collated = collator.collate(examples)
        input_gene_ids = collated["gene"]
        expressions = collated["expr"]
        padding_mask = input_gene_ids.eq(vocab[pad_token])
        return input_gene_ids.long(), expressions.float(), padding_mask.bool()

    @staticmethod
    def _extract_embeddings(emb_dict, exp_data):
        matching_names = [name for name in exp_data.columns if name in emb_dict]
        if not matching_names:
            raise ValueError("No protein names in the expression table match the embedding dictionary")
        missing_names = [name for name in exp_data.columns if name not in emb_dict]
        if missing_names:
            exp_data = exp_data.drop(columns=missing_names)
        embeddings = np.vstack([emb_dict[name] for name in matching_names])
        return embeddings, exp_data, exp_data.columns.tolist()

    def _validate_static_features(self):
        required = {"image_paths", "hist_features", "hist_celltype_prob", "pred_rna_emb", "pred_prot_emb"}
        missing = required - set(self.static_features)
        if missing:
            raise ValueError(f"Static feature cache is missing keys: {sorted(missing)}")
        cache_paths = list(self.static_features["image_paths"])
        if cache_paths != self.image_paths.tolist():
            raise ValueError("Static feature cache image_paths do not match the dataset order")
        for key in required - {"image_paths"}:
            if self.static_features[key].shape[0] != len(self.image_paths):
                raise ValueError(f"Static feature {key} has an invalid number of rows")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.image_root / str(self.image_paths[idx])
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            raise FileNotFoundError(f"Unable to open image: {image_path}") from exc
        image = np.asarray(image)
        height, width = image.shape[:2]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"Image {image_path} is smaller than crop_size={self.crop_size}")
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        image = self.non_augmentations(image[start_y : start_y + self.crop_size, start_x : start_x + self.crop_size])
        if self.transform is not None:
            image = self.transform(image)

        result = (
            image,
            self.protein_expression[idx],
            torch.as_tensor(self.protein_emb, dtype=torch.float32),
            self.input_protein_ids[idx],
            self.protein_expressions[idx],
            self.src_key_padding_mask_protein[idx],
        )
        if self.static_features is None:
            return result
        return result + tuple(self.static_features[key][idx].float() for key in (
            "hist_features", "hist_celltype_prob", "pred_rna_emb", "pred_prot_emb"
        ))

    def get_protein_names(self):
        return self.protein_names

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import tifffile
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
import torchvision.transforms as transforms
# from config import CFG
from sklearn.neighbors import NearestNeighbors
import anndata as ad
from scgpt.tokenizer import GeneVocab
# from scgpt import DataCollator
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Mapping, Tuple, Union
from tqdm.auto import tqdm  # Use tqdm.auto for notebook/console compatibility
import logging


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. (Simplified for no DataLoader)
    """
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
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")
        if self.mlm_probability <= 0 or self.mlm_probability >= 1:
            raise ValueError("`mlm_probability` must be between 0 and 1.")
        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

    def collate(
        self, examples: List[Dict[str, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collates a list of examples (simplified for direct numpy array input).
        """
        device = "cpu"  # Determine device

        if not isinstance(examples[0], Mapping):
          raise  NotImplementedError

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        padded_genes = []
        padded_expressions = []

        for example in examples:
            genes = torch.from_numpy(example["genes"]).long()
            expressions = torch.from_numpy(example["expressions"]).float()
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=51,
                )

            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)


        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
        }

        if self.do_mlm:
            masked_expressions = self._mask(padded_expressions)
        else:
            masked_expressions = padded_expressions

        data_dict["masked_expr"] = masked_expressions
        return data_dict



    def _mask(self, expressions: torch.Tensor) -> torch.Tensor:
        """Masks expression values."""
        probability_matrix = torch.full(expressions.shape, self.mlm_probability)
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0
        mask = torch.bernoulli(probability_matrix).bool()
        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            return genes[:max_length], expressions[:max_length]
        return self._pad(genes, expressions, max_length)

    def _sample(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes))[:max_length]
            return genes[indices], expressions[indices]
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices]

    def _pad(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
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
        return genes, expressions


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


class ProteinExpressionDataset(Dataset):
    def __init__(self, dataframe, scgpt_model_path, transform=None, protein_emb_path=None, static_feature_path=None):
        """
        PyTorch Dataset for loading H&E images and protein expression data.

        Args:
            dataframe (pd.DataFrame): DataFrame where:
                - The first column contains the file paths to H&E JPEG images.
                - The remaining columns contain protein expression values.
                - The column names (after the first column) are protein names.
            transform (callable, optional): Transformation to be applied to H&E images.
            protein_emb_path (str, optional): Path to the pre-trained protein embeddings.
            static_feature_path (str, optional): Path to pre-computed static features cache (.pt).
        """
        self.dataframe = dataframe
        self.image_paths = dataframe.iloc[:, 0].values  # JPEG image file paths
        self.protein_expression = dataframe.iloc[:, 1:]  # Protein expression values
        self.protein_names = dataframe.columns[1:]  # Protein column names
        self.transform = transform
        self.protein_emb = None
        self.non_augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.scgpt_model_dir = Path(scgpt_model_path)

        # Load protein embeddings if protein_emb_path is provided
        if protein_emb_path:
            protein_embedding = torch.load(protein_emb_path)  # Load pre-trained embeddings
            # Process embeddings to align with the protein names in DataFrame
            self.protein_emb, self.protein_exp, self.protein_names = self._extract_embeddings(protein_embedding, self.protein_expression)

        self.input_protein_ids, self.protein_expressions, self.src_key_padding_mask_protein = self._constract_scgpt_input(self.protein_exp, self.scgpt_model_dir)
        self.protein_expression = self.protein_exp.values
        self.protein_expression = np.log1p(self.protein_expression)

        # Load static features cache if provided
        self.static_features = None
        if static_feature_path is not None:
            self.static_features = torch.load(static_feature_path, map_location="cpu")
            self._validate_static_features()


    def _constract_scgpt_input(self, input_count_matrix, scgpt_model_dir):
        vocab_file = scgpt_model_dir / "vocab.json"
        model_config_file = scgpt_model_dir / "args.json"
        model_file = scgpt_model_dir / "best_model.pt"
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        # vocabulary
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        adata = ad.AnnData(input_count_matrix)
        adata.var['gene'] = adata.var.index.tolist()
        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata.var['gene']
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.
        vocab.set_default_index(vocab["<pad>"])
        genes = adata.var['gene'].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        count_matrix = adata.X
        count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
        )  # Ensure numpy array
        # 1. Prepare the data (similar to the Dataset class)
        all_examples = []
        for i in range(count_matrix.shape[0]):
            row = count_matrix[i]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = gene_ids[nonzero_idx]
            # Prepend <cls> token
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs.get("pad_value", 0))  # Use get for safety
            all_examples.append({"genes": genes, "expressions": values})
        
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs.get("pad_token", "<pad>")],  # Use get with a default
            pad_value=model_configs.get("pad_value", 0),
            do_mlm=False,  # Set to True if you want MLM
            do_binning=True,
            max_length=900,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collated_data = collator.collate(all_examples)
        input_gene_ids = collated_data["gene"]
        expressions = collated_data["expr"]
        src_key_padding_mask = input_gene_ids.eq(vocab[model_configs.get("pad_token","<pad>")])

        # 转换为 torch.Tensor
        input_gene_ids = torch.tensor(input_gene_ids, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float32)
        src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)

        return input_gene_ids, expressions, src_key_padding_mask


    def _extract_embeddings(self, emb_dict, exp_data):
        """
        Extract and align protein embeddings with the columns (proteins) in the expression data.

        Args:
            emb_dict (dict): Dictionary of embeddings keyed by protein names.
            exp_data (np.ndarray): Protein expression data from DataFrame.

        Returns:
            Tuple:
                - final_embeddings_array (np.ndarray): Extracted protein embeddings as a stacked array.
                - exp_data (np.ndarray): Updated protein expression data (aligned with embeddings).
                - keys_to_extract (list): List of protein names that match embeddings.
        """
        keys_to_extract = exp_data.columns.tolist()  # Protein names in the DataFrame
        extracted_embeddings = []
        keys_missing = []
        
        for key in keys_to_extract:
            if key in emb_dict:
                # Append the embedding for the current protein
                extracted_embeddings.append(emb_dict[key])
            else:
                # Keep track of missing keys
                keys_missing.append(key)

        # Stack embeddings into a single numpy array
        if not extracted_embeddings:
            raise ValueError("No matching protein embeddings found in the embedding dictionary.")
        
        final_embeddings_array = np.vstack(extracted_embeddings)

        # Remove missing proteins from expression data
        if keys_missing:
            exp_data = exp_data.drop(columns=keys_missing)

        return final_embeddings_array, exp_data, exp_data.columns.tolist()

    def _validate_static_features(self):
        """Validate that the static features cache matches the dataset.

        Checks:
          1. All required keys are present.
          2. Number of image paths matches the dataset.
          3. Image path ordering is identical (raises ValueError on mismatch).
          4. All tensors have matching first dimension.
        """
        required_keys = {
            "image_paths",
            "hist_features",
            "hist_celltype_prob",
            "pred_rna_emb",
            "pred_prot_emb",
        }
        cache = self.static_features
        missing = required_keys - set(cache.keys())
        if missing:
            raise ValueError(f"Static features cache missing keys: {missing}")

        cache_paths = cache["image_paths"]
        if len(cache_paths) != len(self.image_paths):
            raise ValueError(
                f"Cache has {len(cache_paths)} entries but dataset has "
                f"{len(self.image_paths)} entries."
            )

        # Verify order consistency — mismatch would silently pair wrong features
        if cache_paths != self.image_paths.tolist():
            raise ValueError(
                "Static features cache image_paths order does not match "
                "dataset image_paths order."
            )

        n = len(self.image_paths)
        for key in ["hist_features", "hist_celltype_prob", "pred_rna_emb", "pred_prot_emb"]:
            t = cache[key]
            if t.shape[0] != n:
                raise ValueError(
                    f"Static feature '{key}' has {t.shape[0]} rows, expected {n}."
                )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple:
                - image (numpy.ndarray): H&E image as a numpy array.
                - protein_expression (numpy.ndarray): Protein expression values as a 1D numpy array.
                - protein_emb (torch.Tensor): Pre-trained protein embedding tensor.
        """
        # Load the image
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path) 
        new_filename = os.path.splitext(filename)[0]  + ".tif"
        new_path = os.path.join("new_he",  new_filename)
        try:
            image = Image.open(os.path.join('/macroverse-nas/pjz/crc_codex/codex/codex/ORIONCRC_dataset_tile_20x/', image_path)).convert('RGB')  # Ensure images are RGB format
        except Exception as e:
            print(f"Warning: failed to open image {new_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # 或者 raise 或其他处理方式
        image = np.array(image)  # Convert to numpy array
    
        height, width, channels = image.shape  # (333, 333, 3)

        crop_size = 256
        start_x = (width - crop_size) // 2  # 水平方向中心点起始位置
        start_y = (height - crop_size) // 2  # 垂直方向中心点起始位置
        image_cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
        image_cropped = self.non_augmentations(image_cropped)

        # Apply optional image transformations (if provided)
        if self.transform:
            image_cropped = self.transform(image_cropped)

        input_protein_ids = self.input_protein_ids[idx]
        protein_expressions = self.protein_expressions[idx]
        src_key_padding_mask_protein = self.src_key_padding_mask_protein[idx]

        # Load the protein expression values for the current Spot
        protein_expression = self.protein_expression[idx].astype(np.float32)

        # Return image, protein expression, and protein embedding
        if self.static_features is not None:
            hist_features = self.static_features["hist_features"][idx].float()
            hist_celltype_prob = self.static_features["hist_celltype_prob"][idx].float()
            pred_rna_emb = self.static_features["pred_rna_emb"][idx].float()
            pred_prot_emb = self.static_features["pred_prot_emb"][idx].float()
            return (
                image_cropped,
                protein_expression,
                torch.tensor(self.protein_emb, dtype=torch.float32),
                input_protein_ids,
                protein_expressions,
                src_key_padding_mask_protein,
                hist_features,
                hist_celltype_prob,
                pred_rna_emb,
                pred_prot_emb,
            )
        return image_cropped, protein_expression, torch.tensor(self.protein_emb, dtype=torch.float32), input_protein_ids, protein_expressions, src_key_padding_mask_protein

    def get_protein_names(self):
        return self.protein_names
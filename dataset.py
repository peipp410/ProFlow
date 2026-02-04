# -*- coding: utf-8 -*-
import os
# import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from skimage.io import imread, imshow
# from skimage.measure import regionprops
# from skimage.util import crop
# from csbdeep.utils import normalize
# from stardist.models import StarDist2D
# from shapely.geometry import Polygon, Point
# from scipy.sparse import csr_matrix
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

# Setting logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        print(bins)
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


class BulkDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 rna_path,
                 protein_path,
                 sample_group_path,
                 scgpt_model_path,
                 protein_emb_path):
        self.rna_exp = pd.read_parquet(rna_path)
        self.rna_exp = self.rna_exp.T
        self.barcode_tsv = self.rna_exp.index.tolist()

        protein_embedding = torch.load(protein_emb_path)

        self.protein_exp = pd.read_csv(protein_path, sep=",", header=0, index_col=0)  # features * cell
        missing_rate = self.protein_exp.isnull().mean(axis=1) 
        self.protein_exp = self.protein_exp[missing_rate < 0.1]
        self.protein_exp = self.protein_exp.fillna(0)
        self.protein_exp = self.protein_exp.T

        # self.protein_exp = self.protein_exp.dropna(axis=1, how='any')
        # self.protein_exp = (self.protein_exp - self.protein_exp.mean(axis=0)) / self.protein_exp.std(axis=0)

        self.protein_embedding, self.protein_exp, self.protein_names = self._extract_embeddings_to_numpy(protein_embedding, self.protein_exp)

        print(self.protein_exp.shape)
        all_zeros = (self.rna_exp.abs() < 1e-9).all(axis=1)
        self.zero_rows_index = all_zeros[all_zeros].index.tolist()
        if self.zero_rows_index:  # Check zero_rows_index 
            self.rna_exp.loc[self.zero_rows_index, self.rna_exp.columns[:2]] = 1

        self.scgpt_model_dir = Path(scgpt_model_path)

        self.sample_group = pd.read_csv(sample_group_path, index_col=0)
        self.sample_group = self.sample_group.iloc[:, 0]
        if self.sample_group.dtype == 'object':  
            self.sample_group = self.sample_group.astype('category')  
            self.sample_group = self.sample_group.cat.codes  
        self.sample_group = self.sample_group.to_numpy()


        # Preprocess scGPT input
        self.input_gene_ids, self.rna_expressions, self.src_key_padding_mask = self._constract_scgpt_input(self.rna_exp, self.scgpt_model_dir)
        self.input_protein_ids, self.protein_expressions, self.src_key_padding_mask_protein = self._constract_scgpt_input(self.protein_exp, self.scgpt_model_dir)

        self.failed_indices = []
        # for _ in self.zero_rows_index:
        #     self.failed_indices.append(_) 

    def _extract_embeddings_to_numpy(self, emb_dict, exp_df):

        keys_to_extract = exp_df.columns.tolist()
    
        if not keys_to_extract:
            print("Warning: No keys provided to extract.")
            return np.array([]) # Return an empty array if the input list is empty

        extracted_embeddings = []
        keys_found = []
        keys_missing = []

        try:
            for key in keys_to_extract:
                if key in emb_dict:
                    try:
                        # Read the entire dataset associated with the key
                        # f[key][:] reads the data into a numpy array
                        embedding_data = emb_dict[key]
                        extracted_embeddings.append(embedding_data)
                        keys_found.append(key)
                        # print(f"Successfully extracted data for key: '{key}' (shape: {embedding_data.shape})")
                    except Exception as e:
                        print(f"Error reading data for key '{key}': {e}")
                        keys_missing.append(key) # Treat read error as missing
                else:
                    # print(f"Warning: Key '{key}' not found in HDF5 file.")
                    keys_missing.append(key)

            # Check if any embeddings were successfully extracted
            if not extracted_embeddings:
                print("Error: No valid embeddings found for the specified keys.")
                return None

            # Stack the embeddings into a single NumPy array
            # np.vstack assumes each embedding is a 1D array (vector) and stacks them row-wise.
            # If your embeddings are multi-dimensional and you want to concatenate them
            # along the first axis, np.array(extracted_embeddings) or np.concatenate might be better,
            # but ensure shapes are compatible. vstack is common for lists of 1D embeddings.
            try:
                final_embeddings_array = np.vstack(extracted_embeddings)
                print(f"\nSuccessfully stacked {len(keys_found)} embeddings.")
                print(f"Final array shape: {final_embeddings_array.shape}")
                if keys_missing:
                    print(f"Keys not found or failed to read: {keys_missing}")
                    exp_df = exp_df.drop(labels=keys_missing, axis=1, inplace=False, errors='ignore')
                
                return final_embeddings_array, exp_df, exp_df.columns.to_list()
            except ValueError as e:
                print(f"\nError stacking embeddings: {e}")
                print("This might happen if the embeddings have incompatible shapes for vstack.")
                # Provide more info if possible:
                shapes = [emb.shape for emb in extracted_embeddings]
                print(f"Shapes of extracted embeddings: {shapes}")
                return None # Or handle differently, e.g., return the list itself
            except Exception as e:
                print(f"\nAn unexpected error occurred during stacking: {e}")
                return None

        except Exception as e:
            print(f"An error occurred while accessing the HDF5 file: {e}")
            return None

    def get_protein_names(self):
        return self.protein_names

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
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collated_data = collator.collate(all_examples)
        input_gene_ids = collated_data["gene"]
        expressions = collated_data["expr"]
        src_key_padding_mask = input_gene_ids.eq(vocab[model_configs.get("pad_token","<pad>")])

        # Convert to torch.Tensor
        input_gene_ids = torch.tensor(input_gene_ids, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float32)
        src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)

        return input_gene_ids, expressions, src_key_padding_mask
    
    def __getitem__(self, idx):

        actual_idx = idx

        if actual_idx in self.failed_indices:
          return None 

        item = {}
        barcode = self.barcode_tsv[actual_idx]

        if barcode in self.zero_rows_index:
            logging.warning(f"Barcode {barcode} is in zero rows index. Skipping.")
            self.failed_indices.append(idx) 
            return None

        input_gene_ids = self.input_gene_ids[actual_idx]
        rna_expressions = self.rna_expressions[actual_idx]
        src_key_padding_mask = self.src_key_padding_mask[actual_idx]
        input_protein_ids = self.input_protein_ids[actual_idx]
        protein_expressions = self.protein_expressions[actual_idx]
        src_key_padding_mask_protein = self.src_key_padding_mask_protein[actual_idx]

        item['input_gene_ids'] = input_gene_ids
        item['rna_expressions'] = rna_expressions
        item['src_key_padding_mask'] = src_key_padding_mask
        item['input_protein_ids'] = input_protein_ids
        item['protein_expressions'] = protein_expressions
        item['src_key_padding_mask_protein'] = src_key_padding_mask_protein
        item['protein_emb'] = torch.tensor(self.protein_embedding, dtype=torch.float32)
        item['protein_exp_ori'] = torch.tensor(self.protein_exp.values[actual_idx], dtype=torch.float32)
        item['sample_group'] = self.sample_group[actual_idx]

        return item
    
    def __len__(self):
        return len(self.barcode_tsv)
    


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 rna_path,
                 sample_group_path,
                 scgpt_model_path,
                 protein_emb_path,
                 protein_names):
        self.rna_exp = pd.read_parquet(rna_path)
        self.rna_exp = self.rna_exp.T
        self.barcode_tsv = self.rna_exp.index.tolist()

        protein_embedding = torch.load(protein_emb_path)

        # self.protein_exp = pd.read_csv(protein_path, sep=",", header=0, index_col=0)  # features * cell
        # missing_rate = self.protein_exp.isnull().mean(axis=1) 
        # self.protein_exp = self.protein_exp[missing_rate < 0.1]
        # self.protein_exp = self.protein_exp.fillna(0)
        # self.protein_exp = self.protein_exp.T

        # # self.protein_exp = self.protein_exp.dropna(axis=1, how='any')
        # # self.protein_exp = (self.protein_exp - self.protein_exp.mean(axis=0)) / self.protein_exp.std(axis=0)

        self.protein_embedding = self._extract_embeddings_to_numpy(protein_embedding, protein_names)

        # print(self.protein_exp.shape)
        all_zeros = (self.rna_exp.abs() < 1e-9).all(axis=1)
        self.zero_rows_index = all_zeros[all_zeros].index.tolist()
        if self.zero_rows_index:  # Check zero_rows_index 
            self.rna_exp.loc[self.zero_rows_index, self.rna_exp.columns[:2]] = 1

        self.scgpt_model_dir = Path(scgpt_model_path)

        self.sample_group = pd.read_csv(sample_group_path, index_col=0)
        self.sample_group = self.sample_group.iloc[:, 0]
        if self.sample_group.dtype == 'object':  
            self.sample_group = self.sample_group.astype('category')  
            self.sample_group = self.sample_group.cat.codes  
        self.sample_group = self.sample_group.to_numpy()


        # Preprocess scGPT input
        self.input_gene_ids, self.rna_expressions, self.src_key_padding_mask = self._constract_scgpt_input(self.rna_exp, self.scgpt_model_dir)
        # self.input_protein_ids, self.protein_expressions, self.src_key_padding_mask_protein = self._constract_scgpt_input(self.protein_exp, self.scgpt_model_dir)

        self.failed_indices = []
        # for _ in self.zero_rows_index:
        #     self.failed_indices.append(_) 

    def _extract_embeddings_to_numpy(self, emb_dict, keys_to_extract):

    
        if not keys_to_extract:
            print("Warning: No keys provided to extract.")
            return np.array([]) # Return an empty array if the input list is empty

        extracted_embeddings = []
        keys_found = []
        keys_missing = []

        try:
            for key in keys_to_extract:
                if key in emb_dict:
                    try:
                        # Read the entire dataset associated with the key
                        # f[key][:] reads the data into a numpy array
                        embedding_data = emb_dict[key]
                        extracted_embeddings.append(embedding_data)
                        keys_found.append(key)
                        # print(f"Successfully extracted data for key: '{key}' (shape: {embedding_data.shape})")
                    except Exception as e:
                        print(f"Error reading data for key '{key}': {e}")
                        keys_missing.append(key) # Treat read error as missing
                else:
                    # print(f"Warning: Key '{key}' not found in HDF5 file.")
                    keys_missing.append(key)

            # Check if any embeddings were successfully extracted
            if not extracted_embeddings:
                print("Error: No valid embeddings found for the specified keys.")
                return None

            # Stack the embeddings into a single NumPy array
            # np.vstack assumes each embedding is a 1D array (vector) and stacks them row-wise.
            # If your embeddings are multi-dimensional and you want to concatenate them
            # along the first axis, np.array(extracted_embeddings) or np.concatenate might be better,
            # but ensure shapes are compatible. vstack is common for lists of 1D embeddings.
            try:
                final_embeddings_array = np.vstack(extracted_embeddings)
                print(f"\nSuccessfully stacked {len(keys_found)} embeddings.")
                print(f"Final array shape: {final_embeddings_array.shape}")
                
                return final_embeddings_array
            except ValueError as e:
                print(f"\nError stacking embeddings: {e}")
                print("This might happen if the embeddings have incompatible shapes for vstack.")
                # Provide more info if possible:
                shapes = [emb.shape for emb in extracted_embeddings]
                print(f"Shapes of extracted embeddings: {shapes}")
                return None # Or handle differently, e.g., return the list itself
            except Exception as e:
                print(f"\nAn unexpected error occurred during stacking: {e}")
                return None

        except Exception as e:
            print(f"An error occurred while accessing the HDF5 file: {e}")
            return None

    def get_protein_names(self):
        return self.protein_names

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
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collated_data = collator.collate(all_examples)
        input_gene_ids = collated_data["gene"]
        expressions = collated_data["expr"]
        src_key_padding_mask = input_gene_ids.eq(vocab[model_configs.get("pad_token","<pad>")])

        # Convert to torch.Tensor
        input_gene_ids = torch.tensor(input_gene_ids, dtype=torch.long)
        expressions = torch.tensor(expressions, dtype=torch.float32)
        src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)

        return input_gene_ids, expressions, src_key_padding_mask
    
    def __getitem__(self, idx):

        actual_idx = idx

        if actual_idx in self.failed_indices:
          return None 

        item = {}
        barcode = self.barcode_tsv[actual_idx]
        item['barcode'] = barcode

        if barcode in self.zero_rows_index:
            logging.warning(f"Barcode {barcode} is in zero rows index. Skipping.")
            self.failed_indices.append(idx) 
            return None

        input_gene_ids = self.input_gene_ids[actual_idx]
        rna_expressions = self.rna_expressions[actual_idx]
        src_key_padding_mask = self.src_key_padding_mask[actual_idx]
        # input_protein_ids = self.input_protein_ids[actual_idx]
        # protein_expressions = self.protein_expressions[actual_idx]
        # src_key_padding_mask_protein = self.src_key_padding_mask_protein[actual_idx]

        item['input_gene_ids'] = input_gene_ids
        item['rna_expressions'] = rna_expressions
        item['src_key_padding_mask'] = src_key_padding_mask
        # item['input_protein_ids'] = input_protein_ids
        # item['protein_expressions'] = protein_expressions
        # item['src_key_padding_mask_protein'] = src_key_padding_mask_protein
        item['protein_emb'] = torch.tensor(self.protein_embedding, dtype=torch.float32)
        # item['protein_exp_ori'] = torch.tensor(self.protein_exp.values[actual_idx], dtype=torch.float32)
        item['sample_group'] = self.sample_group[actual_idx]

        return item
    
    def __len__(self):
        return len(self.barcode_tsv)
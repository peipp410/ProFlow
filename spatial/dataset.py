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
from skimage.filters import threshold_li

# 设置日志
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
    def __init__(self, barcode_tsv,
                 image_path,
                 spatial_pos_path,
                 count_mtx_path,
                 protein_exp_path,
                 enrichment_path,
                 scgpt_model_path,
                 protein_emb_path,
                 augment_factor=1):
        # self.whole_image = imread(image_path)
        with tifffile.TiffFile(image_path) as tif:
            self.whole_image = tif.pages[0].asarray()
        if self.whole_image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        self.spatial_pos_csv = pd.read_parquet(spatial_pos_path)
        # self.spatial_pos_csv = self.spatial_pos_csv.reset_index(drop=True)
        self.barcode_tsv = barcode_tsv

        self.count_matrix = pd.read_parquet(count_mtx_path, engine='pyarrow')  # features * cell
        # self.count_matrix = self.count_matrix.reset_index(drop=True)
        # self.genes = self.count_matrix.index.values

        self.protein_exp = pd.read_parquet(protein_exp_path)
        # self.protein_exp = self.protein_exp.drop(columns=['CD45RA-1', 'CD45RO-1'])
        self.protein_exp.columns = ['PDCD1', 'VSIR', 'CD274', 'LAG3', 'FCGR3A', 'GZMB', 'CD163', 'CD4', 'MS4A1', 'CD8A', 'CD3E', 'SDC1', 'HLA-DRA', 'ITGAX', 'CD68', 'PTPRCRA', 'PCNA', 'PTPRCRO', 'MKI67', 'CTNNB1', 'PECAM1', 'PTEN', 'KRT19', 'VIM', 'ACTA2', 'PTPRC', 'CDH1']

        # self.protein_exp = np.log1p(self.protein_exp)

        self.enrichment = pd.read_parquet(enrichment_path)
        self.enrichment = self.enrichment.values

        if protein_emb_path:
            protein_embedding = torch.load(protein_emb_path)  # Load pre-trained embeddings
            # Process embeddings to align with the protein names in DataFrame
            self.protein_emb, self.protein_exp, self.protein_names = self._extract_embeddings_to_numpy(protein_embedding, self.protein_exp)


        # self.protein_expression = (
        #     self.protein_expression - np.mean(self.protein_expression, axis=0)
        # ) / np.std(self.protein_expression, axis=0)

        self.augment_factor = augment_factor

        self.scgpt_model_dir = Path(scgpt_model_path)

        all_zeros = (self.count_matrix.abs() < 1e-9).all(axis=1)
        self.zero_rows_index = all_zeros[all_zeros].index.tolist()
        if self.zero_rows_index:  # 检查 zero_rows_index 是否非空
            self.count_matrix.loc[self.zero_rows_index, self.count_matrix.columns[:2]] = 1
        #print("Finished loading all files")
        # self.seg_model = StarDist2D.from_pretrained('2D_versatile_he')
        # self.whole_image_norm = normalize(self.whole_image, 5, 95)
        # self.all_labels, _ = self.seg_model.predict_instances(self.whole_image_norm, prob_thresh=0.2)

        # self.all_labels = np.array(imread(cell_label_path))

        
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=(0, 360)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
            # transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.non_augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.resnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 在 __init__ 中预处理 scGPT 输入
        self.input_gene_ids, self.expressions, self.src_key_padding_mask = self._constract_scgpt_input(self.count_matrix, self.scgpt_model_dir)
        self.input_protein_ids, self.protein_expressions, self.src_key_padding_mask_protein = self._constract_scgpt_input(self.protein_exp, self.scgpt_model_dir)
        # self.protein_expression = (self.protein_exp.values - self.protein_exp.values.mean(axis=0)) / self.protein_exp.values.std(axis=0) ### scale
        self.protein_expression = self.protein_exp.values

        # 预先计算细胞分割结果 (如果GPU内存允许)
        # self.precomputed_segmentations = self._precompute_segmentations()
        self.failed_indices = []
        # for _ in self.zero_rows_index:
        #     self.failed_indices.append(_) #用于存储失败的索引

    def _extract_embeddings_to_numpy(self, emb_dict, exp_data):
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

        # for i in range(count_matrix.shape[0]):
        #     row = count_matrix[i]
        #     genes = gene_ids
        #     values = row
        #     # Prepend <cls> token
        #     genes = np.insert(genes, 0, vocab["<cls>"])
        #     values = np.insert(values, 0, model_configs.get("pad_value", 0))
        #     all_examples.append({"genes": genes, "expressions": values})
        
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

    def transform(self, image, augment_idx):
        image = Image.fromarray(image)
        if augment_idx > 0:
            image = self.augmentations(image)
        else:
            image = self.non_augmentations(image)
        return image


    def __getitem__(self, idx):

        actual_idx = idx
        augment_idx = idx // len(self.barcode_tsv)

        # 如果索引在失败列表中，跳过
        if actual_idx in self.failed_indices:
            return None # 或者返回一个默认的空样本

        item = {}
        barcode = self.barcode_tsv[actual_idx]

        # 检查 barcode 是否在零行索引中
        if barcode in self.zero_rows_index:
            logging.warning(f"Barcode {barcode} is in zero rows index. Skipping.")
            self.failed_indices.append(idx) #将失败的索引添加
            return None

        spatial_info = self.spatial_pos_csv.loc[[barcode]]
        # celltype_info = self.cell_deconv_matrix[actual_idx]

        # 从预处理的数据中获取
        input_gene_ids = self.input_gene_ids[actual_idx]
        expressions = self.expressions[actual_idx]
        src_key_padding_mask = self.src_key_padding_mask[actual_idx]
        input_protein_ids = self.input_protein_ids[actual_idx]
        protein_expressions = self.protein_expressions[actual_idx]
        src_key_padding_mask_protein = self.src_key_padding_mask_protein[actual_idx]

        protein_expression = self.protein_expression[actual_idx].astype(np.float32)


        if spatial_info.empty:
            logging.warning(f"Warning: No spatial information found for barcode {barcode}. Skipping.")
            self.failed_indices.append(idx)
            return None

        try:
            v1 = int(spatial_info['row'].iloc[0])
            v2 = int(spatial_info['col'].iloc[0])
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Error processing barcode {barcode}: {e}. Skipping.")
            self.failed_indices.append(idx)
            return None

        # try:
        #     image_patch_raw = self.whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
        # except IndexError as e:
        #     # logging.error(f"Error extracting image patch for barcode {barcode}: {e}. Likely coordinates are out of bounds. Skipping.")
        #     self.failed_indices.append(idx)
        #     return None

        # Define patch boundaries
        r_start, r_end = v1 - 128, v1 + 128
        c_start, c_end = v2 - 128, v2 + 128

        # Check boundaries
        img_h, img_w = self.whole_image.shape[:2]
        # if r_start < 0 or r_end > img_h or c_start < 0 or c_end > img_w:
        #      logging.warning(f"Patch boundaries out of image bounds for barcode {barcode}. Image data is None.")
        #      # items remain None
        if r_start < 0:
            r_start, r_end = 0, 256
        if r_end > img_h:
            r_start, r_end = img_h - 256, img_h
        if c_start < 0:
            c_start, c_end = 0, 256
        if c_end > img_w:
            c_start, c_end = img_w - 256, img_w

        image_patch_raw = self.whole_image[r_start:r_end, c_start:c_end]
        # labels = self.all_labels[r_start:r_end, c_start:c_end]

        if image_patch_raw.shape[0] != 256 or image_patch_raw.shape[1] != 256:
          # logging.warning(f"image_patch_raw shape is not 256*256. Skipping.")
          self.failed_indices.append(idx)
          return None


        image_patch = self.transform(image_patch_raw, augment_idx)

        # item['image'] = image_patch
        # item['input_gene_ids'] = input_gene_ids
        # item['expressions'] = expressions
        # item['src_key_padding_mask'] = src_key_padding_mask
        # item['protein_expression'] = protein_expression
        # item['protein_emb'] = torch.tensor(self.protein_emb, dtype=torch.float32)

        return image_patch, input_gene_ids, expressions, src_key_padding_mask, input_protein_ids, protein_expressions, src_key_padding_mask_protein, protein_expression, torch.tensor(self.protein_emb, dtype=torch.float32), torch.tensor(self.enrichment[actual_idx], dtype=torch.float32), torch.tensor(actual_idx, dtype=torch.long)
    
    def __len__(self):
        return len(self.barcode_tsv) * self.augment_factor

    def get_protein_names(self):
        return self.protein_names
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import tifffile

class InferenceProteinDataset(Dataset):
    def __init__(self, image_path, coord_path, protein_names, protein_emb_path=None, patch_size=256):
        """
        PyTorch Dataset for Inference (Image + Protein Embedding only).
        
        Args:
            image_path (str): Path to the large TIFF image.
            coord_path (str): Path to a txt file containing coordinates (x, y) for each patch.
            protein_names (list): List of protein names to be predicted (determines the order of embeddings).
            transform (callable, optional): Extra transformation to be applied to H&E images.
            protein_emb_path (str): Path to the pre-trained protein embeddings (dictionary).
            patch_size (int): Size of the patch to crop (default 256).
        """
        self.image_path = image_path
        self.patch_size = patch_size

        with tifffile.TiffFile(image_path) as tif:
            self.whole_image = tif.pages[0].asarray()
        
        self.non_augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.coords = self._load_coords(coord_path)

        if not protein_emb_path:
            raise ValueError("protein_emb_path is required for inference.")
            
        raw_emb = torch.load(protein_emb_path, map_location='cpu')
        self.protein_emb, self.final_protein_names = self._extract_embeddings(raw_emb, protein_names)
        
        self.protein_emb_tensor = torch.tensor(self.protein_emb, dtype=torch.float32)

    def _load_coords(self, coord_path):

        coords = []
        with open(coord_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                if len(parts) >= 2:
                    coords.append((int(float(parts[0])), int(float(parts[1]))))
        print(f"Loaded {len(coords)} patch coordinates from {coord_path}")
        return coords

    def _extract_embeddings(self, emb_dict, target_names):

        extracted = []
        valid_names = []
        for name in target_names:
            if name in emb_dict:
                extracted.append(emb_dict[name])
                valid_names.append(name)
            else:
                print(f"Warning: Protein '{name}' not found in embedding file, skipping.")
        
        if not extracted:
            raise ValueError("No matching protein embeddings found.")

        return np.vstack(extracted), valid_names

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        cx, cy = self.coords[idx] 

        start_y = cy - 128  # Row start
        start_x = cx - 128  # Col start
        end_y = start_y + 256
        end_x = start_x + 256

        img_h, img_w = self.whole_image.shape[:2]

        if start_y < 0:
            start_y, end_y = 0, 256
        if end_y > img_h:
            start_y, end_y = img_h - 256, img_h
        if start_x < 0:
            start_x, end_x = 0, 256
        if end_x > img_w:
            start_x, end_x = img_w - 256, img_w

        image_patch_raw = self.whole_image[start_y:end_y, start_x:end_x]
        
        image_patch = self.non_augmentations(image_patch_raw)

        return image_patch, self.protein_emb_tensor, cx, cy

    def get_protein_names(self):
        return self.final_protein_names
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from lora_pytorch import LoRA
from attn_lora import MHAttenLoRA
from itertools import cycle

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
import logging

from accelerate import Accelerator  # Import Accelerator
from accelerate import FullyShardedDataParallelPlugin

torch.autograd.set_detect_anomaly(True)
seed = 42
torch.manual_seed(seed)     
np.random.seed(seed)      
torch.cuda.manual_seed(seed)        
torch.cuda.manual_seed_all(seed)    
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 

logging.basicConfig(
    level=logging.INFO,  # 璁剧疆鏃ュ織绾у埆涓� INFO
    format="%(asctime)s [%(levelname)s] %(message)s",  # 鏍煎紡鍖栨棩蹇楄緭鍑�
    datefmt="%Y-%m-%d %H:%M:%S"  # 璁剧疆鏃堕棿鏍煎紡
)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),  # Use BatchNorm1d for 1D data
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_features, out_features, num_residual_blocks=3):  # Reduced residual blocks
        super(Generator, self).__init__()

        layers = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),  # Use BatchNorm1d
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(out_features))

        layers.append(nn.Linear(out_features, in_features)) # Output back to original dim
        # No final activation.  Good for general feature transformations.
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, 256), # Reduced hidden layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
#            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.model(x)


class HistCycleRNA(nn.Module):
    def __init__(self, rna_model, hist_model, class_embeddings, feature_dim=512):
        super(HistCycleRNA, self).__init__()

        # Initialzation
        self.rna_model = rna_model
        self.hist_model = hist_model
        class_embeddings_tensor = torch.from_numpy(class_embeddings).float()
        self.n_ct = class_embeddings_tensor.shape[0]
        # class_embeddings_tensor = F.normalize(class_embeddings_tensor, dim=-1)
        self.register_buffer('class_embeddings', class_embeddings_tensor)
        # self.trans_dim = nn.Linear(2 * feature_dim, feature_dim)
        self.image_prob = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),  
            nn.Linear(128, self.n_ct, bias=False), 
            nn.Softmax(dim=1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.image_prob.apply(init_weights)

        # Generators, A=RNA, B=Hist
        self.G_AB = Generator(feature_dim, feature_dim)
        self.G_BA = Generator(feature_dim, feature_dim)

        # Discriminators
        self.D_A = Discriminator(feature_dim)
        self.D_B = Discriminator(feature_dim)

        # Gradient penalty lambda
        self.lambda_gp = 10

    def forward(self, image):

        # input_gene_ids = self.rna_model.module.encoder(input_gene_ids)
        # expressions = self.rna_model.module.value_encoder(expressions)
        # total_embs = input_gene_ids + expressions
        # cell_embeddings = self.rna_model.module.transformer_encoder(
        #     total_embs, src_key_padding_mask=src_key_padding_mask
        # )

        # cell_embeddings = cell_embeddings[:, 0, :] # get the <cls> position embedding
        # # cell_embeddings = F.normalize(cell_embeddings, dim=-1)

        # --- Hist embedding(B_1) ---
        image_latent, tokens_embs = self.hist_model.module(image)
        # image_latent = F.normalize(image_latent, dim=-1)

        hist_celltype_prob = self.image_prob(image_latent)

        
        hist_features = torch.matmul(hist_celltype_prob, self.class_embeddings)

        # hist_features = hist_features.detach()
        # cell_embeddings = cell_embeddings.detach()
        
        # --- CycleGAN Part ---
        # fake_hist = self.G_AB(cell_embeddings)
        # recovered_cell = self.G_BA(fake_hist)
        fake_cell = self.G_BA(hist_features)
        # recovered_hist = self.G_AB(fake_cell)

        # # --- Prediction Part ---
        # combined_features = torch.cat((cell_embeddings, hist_features), dim=1)
        # celltypes_pred = self.predictor(combined_features)

        return image_latent, hist_features, fake_cell


class ProtCycleRNA(nn.Module):
    def __init__(self, rna_model, protein_model, feature_dim=512):
        super(ProtCycleRNA, self).__init__()

        # Initialzation
        self.rna_model = rna_model
        self.protein_model = protein_model

        # Generators, A=RNA, B=Protein
        self.G_AB = Generator(feature_dim, feature_dim)
        self.G_BA = Generator(feature_dim, feature_dim)
        # for param in self.G_BA.parameters():
        #     param.requires_grad = False

        # Discriminators
        self.D_A = Discriminator(feature_dim)
        self.D_B = Discriminator(feature_dim)

        # Gradient penalty lambda
        self.lambda_gp = 10

    def forward(self, cell_embeddings_rna):

        # cell_embeddings_rna = self.rna_embedding(input_gene_ids_b, rna_expressions_b, src_key_padding_mask_b)
        # # cell_embeddings = F.normalize(cell_embeddings, dim=-1)

        # input_protein_ids = self.protein_model.module.encoder(input_protein_ids)
        # protein_expressions = self.protein_model.module.value_encoder(protein_expressions)
        # protein_embs = input_protein_ids + protein_expressions
        # cell_embeddings_protein = self.protein_model.module.transformer_encoder(
        #     protein_embs, src_key_padding_mask=src_key_padding_mask_protein
        # )
        # cell_embeddings_protein = cell_embeddings_protein[:, 0, :] # get the <cls> position embedding
        
        # --- CycleGAN Part ---
        fake_protein = self.G_AB(cell_embeddings_rna)
        # recovered_rna = self.G_BA(fake_protein)
        # fake_rna = self.G_BA(cell_embeddings_protein)
        # recovered_protein = self.G_AB(fake_rna)

        return fake_protein
    

class HistFlowRNA(nn.Module):
    def __init__(self, rna_model, hist_model, class_embeddings, flow_model,
                 feature_dim=512, lambda_focal=1.0,
                 lambda_cos=1.0, lambda_mmd=1.0, mmd_kernel=None):
        super().__init__()
        self.rna_model = rna_model
        self.hist_model = hist_model
        self.flow_model = flow_model
        self.lambda_focal = lambda_focal
        self.lambda_cos = lambda_cos
        self.lambda_mmd = lambda_mmd

        # Store MMD kernel
        self.mmd_loss_fn = mmd_kernel

        # Register class embeddings
        class_embeddings_tensor = torch.from_numpy(class_embeddings).float()
        self.n_ct = class_embeddings_tensor.shape[0]
        self.register_buffer('class_embeddings', class_embeddings_tensor)

        # Image classifier head
        self.image_prob = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.n_ct, bias=False),
            # nn.Softmax(dim=1)
            nn.LogSoftmax(dim=1)
        )
        self.image_prob.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def masked_focal_loss(hist_celltype_prob, celltype_mask, gamma=2, alpha=0.25):
        hist_celltype_prob = torch.clamp(hist_celltype_prob, min=1e-7, max=1.0 - 1e-7)
        ce_loss = -torch.log(1 - hist_celltype_prob) * (1 - celltype_mask)
        p_t = (1 - hist_celltype_prob) * (1 - celltype_mask)
        focal_weight = alpha * (1 - p_t) ** gamma
        loss = (focal_weight * ce_loss).sum()
        return loss

    def forward(self, image, input_gene_ids, expressions, src_key_padding_mask,
                celltype_mask=None, steps=60, cond_vec=None):
        cell_embeddings = self.rna_model._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None
        )
        cell_embeddings = cell_embeddings[:, 0, :]

        hist_features, hist_celltype_prob = self._encode_image_features(image)

        pred_cell_emb = self._flow_match(hist_features, steps, cond_vec)

        return pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob

    def _encode_image_features(self, image):
        image_latent, _ = self.hist_model(image)
        hist_celltype_prob_log = self.image_prob(image_latent)  # [B, n_ct]
        hist_celltype_prob = torch.exp(hist_celltype_prob_log)
        hist_features = torch.matmul(hist_celltype_prob, self.class_embeddings)  # [B, feat_dim]
        return image_latent, hist_features, hist_celltype_prob

    def _flow_match(self, hist_features, steps=60, cond_vec=None):
        dt = 1.0 / steps
        z = hist_features.clone()
        for step in range(steps):
            t_val = step / steps
            t_tensor = torch.full((z.size(0), 1), t_val, device=z.device)
            k1 = self.flow_model(z, t_tensor, cond_vec)
            k2 = self.flow_model(z + 0.5 * dt * k1, t_tensor + 0.5 * dt, cond_vec)
            k3 = self.flow_model(z + 0.5 * dt * k2, t_tensor + 0.5 * dt, cond_vec)
            k4 = self.flow_model(z + dt * k3, t_tensor + dt, cond_vec)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return z

    def encode_image_to_rna_emb(self, image, steps=60):
        image_latent, hist_features, hist_celltype_prob = self._encode_image_features(image)
        # image_latent = F.normalize(image_latent, dim=-1)
        pred_cell_emb = self._flow_match(hist_features, steps=steps, cond_vec=None)
        return image_latent, pred_cell_emb, hist_features, hist_celltype_prob
    

class ConditionalFlowNet(nn.Module):
    def __init__(self, source_dim, to_dim, cond_dim, hidden_dim=1024):
        super().__init__()
        self.cond_dim = cond_dim
        input_dim = source_dim + cond_dim + 1  # +1 for t
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, to_dim)
        )

    def forward(self, z_t, t, cond_vec=None):
        """
        z_t: (B, prot_dim)
        t: (B, 1)
        cond_vec: (B, rna_dim + cond_dim)
        """
        if self.cond_dim > 0:
            ht = torch.cat([z_t, cond_vec, t], dim=-1)
        else:
            ht = torch.cat([z_t, t], dim=-1)
        return self.net(ht)


def load_pretrained_model(model, pretrained_weights_path, device):
    # --- Load Pre-trained Weights ---
    try:
        logging.info(f"Loading pre-trained weights from: {pretrained_weights_path}")
        # Load the state dictionary, mapping to the desired device
        # state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
        state_dict = torch.load(pretrained_weights_path, map_location=device)

        # Check if the state_dict is nested (common practice)
        # Adjust these keys ('model_state_dict', 'state_dict') if your saving format was different
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            logging.info("Extracted weights from 'model_state_dict' key.")
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            logging.info("Extracted weights from 'state_dict' key.")
        # Add more checks if needed

        # Load the weights into the model
        # Use strict=False first to diagnose missing/unexpected keys
        logging.info("Attempting to load weights into the model structure...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)

        if missing_keys:
            logging.warning(f"Weights not found for some keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Found unexpected keys in the state dictionary: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            logging.info("Weights loaded successfully (strict match).")
            # You could potentially reload with strict=True here if preferred after verifying no mismatches
            # model.load_state_dict(state_dict, strict=True)
        elif not unexpected_keys and missing_keys:
             logging.warning("Loaded weights successfully, but some model parameters were not in the state_dict (strict=False used).")
        elif not missing_keys and unexpected_keys:
             logging.warning("Loaded weights successfully, but the state_dict contained extra keys not in the model (strict=False used).")
        else:
             logging.warning("Loaded weights with both missing and unexpected keys (strict=False used).")


    except FileNotFoundError:
        logging.error(f"Error: Pre-trained weight file not found at {pretrained_weights_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or applying weights: {e}")
        # Optional: Print keys for debugging mismatch issues
        # print("\nModel state_dict keys (first 10):", list(model.state_dict().keys())[:10])
        # print("\nLoaded state_dict keys (first 10):", list(state_dict.keys())[:10])
        raise

    return model
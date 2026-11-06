import logging

import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


class HistFlowRNA(nn.Module):
    """Extract histology features and map them to the RNA embedding space."""

    def __init__(
        self,
        rna_model,
        hist_model,
        class_embeddings,
        flow_model,
        feature_dim=512,
        lambda_focal=1.0,
        lambda_cos=1.0,
        lambda_mmd=1.0,
    ):
        super().__init__()
        self.rna_model = rna_model
        self.hist_model = hist_model
        self.flow_model = flow_model
        self.lambda_focal = lambda_focal
        self.lambda_cos = lambda_cos
        self.lambda_mmd = lambda_mmd

        class_embeddings_tensor = torch.as_tensor(class_embeddings, dtype=torch.float32)
        self.n_ct = class_embeddings_tensor.shape[0]
        self.register_buffer("class_embeddings", class_embeddings_tensor)

        self.image_prob = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.n_ct, bias=False),
            nn.LogSoftmax(dim=1),
        )
        self.image_prob.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def masked_focal_loss(hist_celltype_prob, celltype_mask, gamma=2, alpha=0.25):
        hist_celltype_prob = torch.clamp(hist_celltype_prob, min=1e-7, max=1.0 - 1e-7)
        ce_loss = -torch.log(1 - hist_celltype_prob) * (1 - celltype_mask)
        p_t = (1 - hist_celltype_prob) * (1 - celltype_mask)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * ce_loss).sum()

    def forward(
        self,
        image,
        input_gene_ids,
        expressions,
        src_key_padding_mask,
        celltype_mask=None,
        steps=60,
        cond_vec=None,
    ):
        cell_embeddings = self.rna_model._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )[:, 0, :]
        _, hist_features, hist_celltype_prob = self._encode_image_features(image)
        pred_cell_emb = self._flow_match(hist_features, steps, cond_vec)
        return pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob

    def _encode_image_features(self, image):
        image_latent, _ = self.hist_model(image)
        hist_celltype_prob_log = self.image_prob(image_latent)
        hist_celltype_prob = torch.exp(hist_celltype_prob_log)
        hist_features = torch.matmul(hist_celltype_prob, self.class_embeddings)
        return image_latent, hist_features, hist_celltype_prob

    def _flow_match(self, hist_features, steps=60, cond_vec=None):
        if steps < 1:
            raise ValueError("steps must be positive")
        dt = 1.0 / steps
        z = hist_features.clone()
        for step in range(steps):
            t = torch.full((z.size(0), 1), step / steps, device=z.device)
            k1 = self.flow_model(z, t, cond_vec)
            k2 = self.flow_model(z + 0.5 * dt * k1, t + 0.5 * dt, cond_vec)
            k3 = self.flow_model(z + 0.5 * dt * k2, t + 0.5 * dt, cond_vec)
            k4 = self.flow_model(z + dt * k3, t + dt, cond_vec)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    def encode_image_to_rna_emb(self, image, steps=60):
        image_latent, hist_features, hist_celltype_prob = self._encode_image_features(image)
        pred_cell_emb = self._flow_match(hist_features, steps=steps)
        return image_latent, pred_cell_emb, hist_features, hist_celltype_prob


class ConditionalFlowNet(nn.Module):
    def __init__(self, source_dim, to_dim, cond_dim, hidden_dim=1024):
        super().__init__()
        self.cond_dim = cond_dim
        input_dim = source_dim + cond_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, to_dim),
        )

    def forward(self, z_t, t, cond_vec=None):
        if self.cond_dim > 0:
            if cond_vec is None:
                raise ValueError("cond_vec is required when cond_dim is positive")
            hidden = torch.cat([z_t, cond_vec, t], dim=-1)
        else:
            hidden = torch.cat([z_t, t], dim=-1)
        return self.net(hidden)


def load_pretrained_model(model, pretrained_weights_path, device):
    LOGGER.info("Loading pre-trained weights from %s", pretrained_weights_path)
    state_dict = torch.load(pretrained_weights_path, map_location=device)
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("model_state_dict", state_dict.get("state_dict", state_dict))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        LOGGER.warning("Missing model weights: %s", missing_keys)
    if unexpected_keys:
        LOGGER.warning("Unexpected model weights: %s", unexpected_keys)
    return model

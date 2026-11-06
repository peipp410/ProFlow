import torch
import torch.nn as nn


class _ResidualMLPBlock(nn.Module):
    """Small pre-normalized residual MLP block for the shared prediction head."""

    def __init__(self, hidden_dim, expansion=2, dropout=0.05, residual_scale=0.1):
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_scale = residual_scale

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return residual + self.residual_scale * x


class PredictModelWithCLasses(nn.Module):
    """Shared ESM-2-conditioned predictor used by the training entry point."""

    def __init__(
        self,
        spot_dim,
        protein_dim,
        num_classes,
        hidden_dim=512,
        num_layers=2,
        dropout=0.05,
        expansion=2,
        residual_scale=0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if expansion < 1:
            raise ValueError("expansion must be at least 1")

        self.protein_projection = nn.Linear(protein_dim, spot_dim)

        self.cell_norm = nn.LayerNorm(spot_dim)
        self.protein_norm = nn.LayerNorm(spot_dim)

        context_dim = spot_dim + num_classes
        fusion_dim = context_dim + spot_dim
        self.input_projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [
                _ResidualMLPBlock(
                    hidden_dim,
                    expansion=expansion,
                    dropout=dropout,
                    residual_scale=residual_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, cell_embeddings_protein, cond_label, protein_emb):
        if cell_embeddings_protein.dim() != 2:
            raise ValueError("cell_embeddings_protein must have shape (B, spot_dim)")
        if cond_label.dim() != 2:
            raise ValueError("cond_label must have shape (B, num_classes)")
        if protein_emb.dim() == 2:
            protein_emb = protein_emb.unsqueeze(0).expand(
                cell_embeddings_protein.size(0), -1, -1
            )
        elif protein_emb.dim() != 3:
            raise ValueError(
                "protein_emb must have shape (P, protein_dim) or " "(B, P, protein_dim)"
            )

        if self.cell_norm is not None:
            cell_embeddings_protein = self.cell_norm(cell_embeddings_protein)

        protein_projected = self.protein_projection(protein_emb)
        n_proteins = protein_projected.size(1)

        if self.protein_norm is not None:
            protein_projected = self.protein_norm(protein_projected)

        context = torch.cat((cell_embeddings_protein, cond_label), dim=-1)
        context = context.unsqueeze(1).expand(-1, n_proteins, -1)
        features = torch.cat((protein_projected, context), dim=-1)

        features = self.input_projection(features)
        for block in self.residual_blocks:
            features = block(features)
        features = self.output_norm(features)
        return self.prediction_head(features).squeeze(-1)

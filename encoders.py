import torch
import torch.nn as nn


class scgpt_encoder(nn.Module):
    def __init__(self, scgpt_model, feature_dim=512):
        super(scgpt_encoder, self).__init__()

        # Initialzation
        self.scgpt_model = scgpt_model

    def forward(self, input_gene_ids, expressions, src_key_padding_mask):

        input_gene_ids = self.scgpt_model.module.encoder(input_gene_ids)
        expressions = self.scgpt_model.module.value_encoder(expressions)
        embs = input_gene_ids + expressions
        cell_embeddings = self.scgpt_model.module.transformer_encoder(
            embs, src_key_padding_mask=src_key_padding_mask
        )
        cell_embeddings = cell_embeddings[:, 0, :] # get the <cls> position embedding

        return cell_embeddings


class PredictModel(nn.Module):
    def __init__(self, spot_dim, protein_dim, num_proteins, feature_dim=512, protein_names=None):
        super(PredictModel, self).__init__()
        self.num_proteins = num_proteins
        self.protein_projection = nn.Linear(protein_dim, spot_dim)

        input_dim = spot_dim
        self.trans_dim = nn.Linear(input_dim * 2, input_dim)
        self.prediction_layers_expression = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim//2),
                nn.LeakyReLU(0.2, inplace=True),  # 
                nn.Linear(input_dim//2, 1)
            ) for _ in range(num_proteins)
        ])

        if protein_names is not None:
            for i, name in enumerate(protein_names):
                self.prediction_layers_expression[i].name = name 

    def forward(self, 
                cell_embeddings_protein,
                protein_emb):

        # 2. Process Protein Data
        protein_projected = self.protein_projection(protein_emb)  # Project protein embeddings

        # 3. Prepare Data for Attention
        spot_encoded = cell_embeddings_protein.unsqueeze(1).repeat(1, self.num_proteins, 1)  # 把cell_embeddings_protein改成fake_protein

        # spot_encoded_norm = spot_encoded / (spot_encoded.norm(dim=-1, keepdim=True) + 1e-8)  # 
        # protein_projected_norm = protein_projected / (protein_projected.norm(dim=-1, keepdim=True) + 1e-8)  # 
        
        # spot_encoded_norm = (spot_encoded - spot_encoded.mean(dim=-1, keepdim=True)) / (spot_encoded.std(dim=-1, keepdim=True) + 1e-8)
        # protein_projected_norm = (protein_projected - protein_projected.mean(dim=-1, keepdim=True)) / (protein_projected.std(dim=-1, keepdim=True) + 1e-8)
 
        # 4. Combine Spot and Protein Features
        # x = spot_encoded_norm + protein_projected_norm # Element-wise addition
        x = torch.cat((protein_projected, spot_encoded), dim=-1)
        x = self.trans_dim(x)

        expression_predictions = []
        for i in range(self.num_proteins):
            protein_embedding = x[:, i, :]  # (N, input_dim)
            expression_prediction = self.prediction_layers_expression[i](protein_embedding)  # (N, 1)
            expression_predictions.append(expression_prediction)

        expression_predictions = torch.cat(expression_predictions, dim=1)

        return expression_predictions


class PredictModelWithCLasses(nn.Module):
    def __init__(self, spot_dim, protein_dim, num_classes, num_proteins, feature_dim=512, protein_names=None):
        super(PredictModelWithCLasses, self).__init__()
        self.num_proteins = num_proteins
        input_dim = spot_dim + num_classes
        self.protein_projection = nn.Linear(protein_dim, spot_dim)

        self.trans_dim = nn.Linear(input_dim + spot_dim, input_dim)
        self.prediction_layers_expression = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim//4),
                nn.LeakyReLU(0.2, inplace=True),  # 
                nn.Linear(input_dim//4, input_dim//2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(input_dim//2, 1)
            ) for _ in range(num_proteins)
        ])

        if protein_names is not None:
            for i, name in enumerate(protein_names):
                self.prediction_layers_expression[i].name = name 

    def forward(self, 
                cell_embeddings_protein,
                cond_label,
                protein_emb):

        # 2. Process Protein Data
        protein_projected = self.protein_projection(protein_emb)  # Project protein embeddings

        spot_encoded = torch.cat((cell_embeddings_protein, cond_label), dim=-1)
        # spot_encoded = cell_embeddings_protein

        # 3. Prepare Data for Attention
        spot_encoded = spot_encoded.unsqueeze(1).repeat(1, self.num_proteins, 1)  # 把cell_embeddings_protein改成fake_protein

        # spot_encoded_norm = spot_encoded / (spot_encoded.norm(dim=-1, keepdim=True) + 1e-8)  # 
        # protein_projected_norm = protein_projected / (protein_projected.norm(dim=-1, keepdim=True) + 1e-8)  # 
        
        # spot_encoded_norm = (spot_encoded - spot_encoded.mean(dim=-1, keepdim=True)) / (spot_encoded.std(dim=-1, keepdim=True) + 1e-8)
        # protein_projected_norm = (protein_projected - protein_projected.mean(dim=-1, keepdim=True)) / (protein_projected.std(dim=-1, keepdim=True) + 1e-8)
 
        # 4. Combine Spot and Protein Features
        # x = spot_encoded_norm + protein_projected_norm # Element-wise addition
        x = torch.cat((protein_projected, spot_encoded), dim=-1)
        x = self.trans_dim(x)

        expression_predictions = []
        for i in range(self.num_proteins):
            protein_embedding = x[:, i, :]  # (N, input_dim)
            expression_prediction = self.prediction_layers_expression[i](protein_embedding)  # (N, 1)
            expression_predictions.append(expression_prediction)

        expression_predictions = torch.cat(expression_predictions, dim=1)

        return expression_predictions
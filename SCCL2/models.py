# script3/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCCLEncoder(nn.Module):
    def __init__(self, base_model, embedding_dim=1024, projection_dim=128, n_clusters=10):
        super(SCCLEncoder, self).__init__()
        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, projection_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, sentences):
        with torch.no_grad():
            embeddings = self.base_model.encode(sentences, convert_to_tensor=True)
        z = self.projection_head(embeddings)
        z = F.normalize(z, dim=1)
        return z

    def get_soft_assignments(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2))
        q = q.pow((1 + 1) / 2)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

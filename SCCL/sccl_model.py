import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, projection_dim=128):  # GTE-large output dim is 1024
        super().__init__()
        self.fc1 = nn.Linear(input_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SCCLEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead()

    def forward(self, sentences):
        with torch.no_grad():
            embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        projected = self.projector(embeddings)
        return F.normalize(projected, dim=-1)

# script3/losses.py

import torch
import torch.nn.functional as F


def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    sim_exp = torch.exp(similarity_matrix / temperature)
    mask = torch.eye(sim_exp.shape[0], dtype=torch.bool).to(sim_exp.device)
    sim_exp = sim_exp.masked_fill(mask, 0)

    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    denom = torch.sum(sim_exp, dim=1)
    loss = -torch.log(pos_sim / denom)
    return loss.mean()


def kl_divergence(q, p):
    return torch.mean(torch.sum(q * torch.log(q / (p + 1e-6)), dim=1))

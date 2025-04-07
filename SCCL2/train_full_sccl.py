# script3/train_full_sccl.py

import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import logging
from sentence_transformers import SentenceTransformer

from models import SCCLEncoder
from losses import contrastive_loss, kl_divergence
from utils import load_utterances

import torch.nn.functional as F

from argparse import ArgumentParser


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.sentences[idx]


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("json_file", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    with open("scripts3/config.json") as f:
        cfg = json.load(f)

    print("📄 Loading data...")
    utterances = load_utterances(args.json_file)

    print("📦 Loading base model...")
    base_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

    model = SCCLEncoder(
        base_model,
        embedding_dim=cfg["embedding_dim"],
        projection_dim=cfg["projection_dim"],
        n_clusters=cfg["n_clusters"]
    )

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    dataset = SentenceDataset(utterances)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    print("🚀 Starting training...")
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch in loader:
            sents_1, sents_2 = batch
            z_i = model(sents_1)
            z_j = model(sents_2)
            q = model.get_soft_assignments(z_i)
            p = target_distribution(q.detach())

            c_loss = contrastive_loss(z_i, z_j, temperature=cfg["temperature"])
            kl_loss = kl_divergence(q, p)
            loss = c_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: total loss = {total_loss:.4f}")

    print("💾 Saving model...")
    torch.save(model.state_dict(), "scripts3/sccl_full_model.pt")
    print("✅ Done!")


if __name__ == "__main__":
    logging.set_verbosity_error()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()

import torch
import json
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sccl_model import SCCLEncoder
from torch.nn.functional import cosine_similarity, normalize, cross_entropy
from tqdm import tqdm

class SCCLDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.sentences[idx]

def nt_xent_loss(z1, z2, temperature=0.1):
    z1, z2 = normalize(z1, dim=1), normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim = cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels])
    sim /= temperature
    return cross_entropy(sim, labels)

def train_sccl_model(sentences, model_path='scripts2/sccl_gte.pt', model_name='Alibaba-NLP/gte-large-en-v1.5', batch_size=64, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = SentenceTransformer(model_name, trust_remote_code=True)
    model = SCCLEncoder(base_model).to(device)
    loader = DataLoader(SCCLDataset(sentences), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sent1, sent2 in tqdm(loader, desc=f"Epoch {epoch+1}"):
            z1, z2 = model(sent1).to(device), model(sent2).to(device)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), model_path)
    print(f"📦 Model saved to {model_path}")

if __name__ == '__main__':
    import sys
    file = sys.argv[1]  # e.g., all.jsonl
    with open(file) as f:
        data = [json.loads(line) for line in f]
    sentences = list({turn['utterance'] for dialog in data for turn in dialog['turns'] if turn['theme_label'] is not None})
    train_sccl_model(sentences)

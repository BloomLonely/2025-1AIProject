from argparse import ArgumentParser
import json
import os
import copy
import collections

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser

from sccl_model import SCCLEncoder
import numpy as np
import torch
import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_file', type=str)
    parser.add_argument('preferences_file', type=str)
    parser.add_argument('result_file', type=str)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--model-path', type=str, default='scripts2/sccl_gte.pt')
    return parser.parse_args()

def find_second_closest_cluster(emb, centroids):
    distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
    sorted_indices = np.argsort(distances)
    return sorted_indices[1]


def apply_preferences_to_clusters(utterances, utterance_embs, cluster_labels, cluster_centroids, shouldlink_pairs, cannot_link_pairs):
    assert len(utterances) == len(cluster_labels)

    datapoint_modification_counter = collections.defaultdict(lambda: 0)

    utterance_cluster_mapping = collections.defaultdict(lambda: -1)
    utterance_idx_mapping = collections.defaultdict(lambda: -1)
    for utt_idx, cluster_label in enumerate(cluster_labels):
        utterance = utterances[utt_idx]
        utterance_cluster_mapping[utterance] = cluster_label
        utterance_idx_mapping[utterance] = utt_idx
    modified_cluster_labels = copy.deepcopy(cluster_labels)
    for utt_a, utt_b in shouldlink_pairs:
        cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
        if cluster_a != cluster_b:
            utt_b_idx = utterance_idx_mapping[utt_b]
            modified_cluster_labels[utt_b_idx] = cluster_a
            utterance_cluster_mapping[utt_b] = cluster_a
            datapoint_modification_counter[utt_b_idx] += 1
    for utt_a, utt_b in cannot_link_pairs:
        cluster_a, cluster_b = utterance_cluster_mapping[utt_a], utterance_cluster_mapping[utt_b]
        if cluster_a == cluster_b:
            utt_b_idx = utterance_idx_mapping[utt_b]
            utt_b_new_cluster = find_second_closest_cluster(utterance_embs[utt_b_idx], cluster_centroids)
            modified_cluster_labels[utt_b_idx] = utt_b_new_cluster
            utterance_cluster_mapping[utt_b] = utt_b_new_cluster
            datapoint_modification_counter[utt_b_idx] += 1
    return modified_cluster_labels

def main(utterances, linking_preferences, sccl_model_path, n_clusters, random_state):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    model = SCCLEncoder(base)
    model.load_state_dict(torch.load(sccl_model_path, map_location=device))
    model.to(device)
    model.eval()

    print("📥 Generating embeddings with SCCL encoder...")
    query_embeddings = []
    for utt in tqdm.tqdm(utterances):
        emb = model([utt]).detach().cpu().numpy()
        query_embeddings.append(emb.squeeze())

    print("📊 Clustering with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(query_embeddings)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    clusters_with_preferences = apply_preferences_to_clusters(
        utterances, query_embeddings, clusters, centroids,
        linking_preferences['should_link'], linking_preferences['cannot_link']
    )

    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, label in enumerate(clusters_with_preferences):
        clustered_utterances[label].append(utterances[i])

    llm = get_llm("mistralai/Mistral-7B-Instruct-v0.3")
    chain = LABEL_CLUSTERS_PROMPT | llm | RunnableParallel(
        theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
        theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
    )

    cluster_label_map = {}
    for i, cluster in enumerate(clustered_utterances):
        out = chain.invoke({'utterances': '\n'.join(cluster)})
        for utt in cluster:
            cluster_label_map[utt] = out['theme_label']['theme_label']
    return cluster_label_map

if __name__ == '__main__':
    args = parse_args()
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf~"

    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]
    themed_utts = list({turn['utterance'] for d in dataset for turn in d['turns'] if turn['theme_label'] is not None})
    
    with open(args.preferences_file) as f:
        prefs = json.load(f)

    pred_map = main(themed_utts, prefs, args.model_path, args.n_clusters, args.random_state)

    updated = copy.deepcopy(dataset)
    for dialog in updated:
        for turn in dialog['turns']:
            if turn['theme_label'] is not None:
                turn['theme_label_predicted'] = pred_map[turn['utterance']]
    with open(args.result_file, 'w') as f:
        for d in updated:
            print(json.dumps(d), file=f)

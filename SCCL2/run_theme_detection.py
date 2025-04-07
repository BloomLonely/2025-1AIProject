import logging

# 로깅 설정 유지
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

import json
import os
import copy
import collections
import tqdm
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch

from langchain_core.runnables import RunnableParallel
from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser
from models import SCCLEncoder


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_file', type=str)
    parser.add_argument('preferences_file', type=str)
    parser.add_argument('result_file', type=str)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--model-path', type=str, default='scripts3/sccl_full_model.pt')
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


def main(utterances, linking_preferences, model_path, n_clusters, random_state):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    model = SCCLEncoder(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logging.info("📥 Generating embeddings with trained SCCL model...")
    query_embeddings = []
    for utt in tqdm.tqdm(utterances):
        with torch.no_grad():
            emb = model([utt]).cpu().numpy()
        query_embeddings.append(emb.squeeze())

    logging.info("📊 Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=100)
    kmeans.fit(query_embeddings)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    logging.info("🔧 Applying preference-based modifications to clustering...")
    clusters_with_preferences = apply_preferences_to_clusters(
        utterances, query_embeddings, clusters, centroids,
        linking_preferences['should_link'], linking_preferences['cannot_link']
    )

    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, label in enumerate(clusters_with_preferences):
        clustered_utterances[label].append(utterances[i])

    llm = get_llm("mistralai/Mistral-7B-Instruct-v0.3")
    chain = (
        LABEL_CLUSTERS_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
            theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
        )
    )

    logging.info("🧠 Generating theme labels using LLM...")
    cluster_label_map = {}
    for i, cluster in enumerate(clustered_utterances):
        logging.info(f"    ▶️ Processing cluster {i + 1}/{n_clusters} ({len(cluster)} utterances)...")
        outputs_parsed = chain.invoke({'utterances': '\n'.join(cluster)})
        predicted_label = outputs_parsed['theme_label']['theme_label']
        for utterance in cluster:
            cluster_label_map[utterance] = predicted_label

    return cluster_label_map


if __name__ == '__main__':
    args = parse_args()

    logging.info("🚀 Starting SCCL-based theme detection pipeline...")
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf~"

    logging.info(f"📂 Loading dataset from: {args.dataset_file}")
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]

    themed_utterances = set()
    for dialogue in dataset:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                themed_utterances.add(turn['utterance'])
    logging.info(f"📊 Total themed utterances: {len(themed_utterances)}")

    logging.info(f"📂 Loading linking preferences from: {args.preferences_file}")
    with open(args.preferences_file) as prefs_in:
        linking_preferences = json.load(prefs_in)

    cluster_label_map = main(
        list(themed_utterances),
        linking_preferences,
        args.model_path,
        args.n_clusters,
        args.random_state
    )

    logging.info("📝 Writing predicted theme labels to result file...")
    dataset_predicted = copy.deepcopy(dataset)
    for dialogue in dataset_predicted:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                turn['theme_label_predicted'] = cluster_label_map[turn['utterance']]

    logging.info(f"📤 Saving result to: {args.result_file}")
    with open(args.result_file, 'w') as result_out:
        for dialogue in dataset_predicted:
            print(json.dumps(dialogue), file=result_out)

    logging.info("✅ Theme detection completed successfully.")


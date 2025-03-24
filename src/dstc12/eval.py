# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import tqdm
import numpy as np
from coclust.evaluation.external import accuracy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from rouge_score.scoring import Score
from langchain_core.runnables import RunnableParallel

from dstc12.utils import DotAllRegexParser
from dstc12.prompts import STYLEGUIDE_SECTION_1_PROMPT, STYLEGUIDE_SECTION_2_PROMPT, STYLEGUIDE_SECTION_3_PROMPT


def acc(references=None, predictions=None):
    assert references and predictions and len(references) == len(predictions)
    return accuracy(references, predictions)


def nmi(references=None, predictions=None):
    assert references and predictions and len(references) == len(predictions)
    return normalized_mutual_info_score(references, predictions)


def rouge_with_multiple_references(references=None, predictions=None, metrics=['rouge1', 'rouge2', 'rougeL'], average=True):
    assert len(references) == len(predictions)
    assert len(set([len(refs_i) for refs_i in references])) == 1, 'Inconsistent number of references per datapoint provided'
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    scores = [scorer.score_multi(refs, pred) for refs, pred in zip(references, predictions)]
    if average:
        result = {}
        for metric in metrics:
            result[metric] = Score(
                np.mean([score[metric].precision for score in scores]),
                np.mean([score[metric].recall for score in scores]),
                np.mean([score[metric].fmeasure for score in scores])
            )
    else:
        result = scores
    return result


def cosine_similarity_with_multiple_references(references_list, predictions):
    similarities = np.zeros((len(references_list), len(predictions)), dtype=np.float32)
    for i, refs_i in enumerate(references_list):
        assert len(refs_i) == len(predictions)
        for j, (ref_ij, pred_j) in enumerate(zip(refs_i, predictions)):
            similarities[i, j] = cosine_similarity(np.array(ref_ij).reshape(1, -1), np.array(pred_j).reshape(1, -1))
    return np.mean(similarities)


def process_llm_judge_output(output):
    scores = []
    for  section in ['section_1', 'section_2', 'section_3']:
        assert section in output and 'score' in output[section]
        scores.append(int(output[section]['score']['value'] == 'Good'))
    return np.mean(scores)


def llm_score(predictions, llm):
    chain = (
        RunnableParallel(
            section_1=STYLEGUIDE_SECTION_1_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
            section_2=STYLEGUIDE_SECTION_2_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
            section_3=STYLEGUIDE_SECTION_3_PROMPT
                | llm
                | RunnableParallel(
                    score=DotAllRegexParser(regex=r'<score>\s*(.*?)\s*</score>', output_keys=['value']),
                    explanation=DotAllRegexParser(regex=r'<explanation>\s*(.*?)\s*</explanation>', output_keys=['value']),
            ),
        )
    )
    scores = []
    for prediction in tqdm.tqdm(predictions):
        judge_output = chain.invoke({'theme_label': prediction})
        scores.append(process_llm_judge_output(judge_output))
    return np.mean(scores)


def ___rouge_with_multiple_references(references_list, predictions):
    scores = [rouge(refs_i, predictions, aggregate=True) for refs_i in references_list]
    scores_averaged = {
        metric_name: {
            'precision': 0,
            'recall': 0,
            'fmeasure': 0
        } for metric_name in scores[0]
    }

    for score_i in scores:
        for metric_name, score in score_i.items():
            scores_averaged[metric_name]['precision'] += score.precision / len(scores)
            scores_averaged[metric_name]['recall'] += score.recall / len(scores)
            scores_averaged[metric_name]['fmeasure'] += score.fmeasure / len(scores)
    return scores_averaged
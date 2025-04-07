# script3/utils.py

import json


def load_utterances(json_file):
    with open(json_file) as f:
        data = [json.loads(line) for line in f]

    utterances = []
    for dialog in data:
        for turn in dialog['turns']:
            if turn.get("theme_label") is not None:
                utterances.append(turn["utterance"])
    return list(set(utterances))

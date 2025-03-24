# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import collections
from typing import Dict
import re

from langchain.output_parsers.regex import RegexParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch


class DotAllRegexParser(RegexParser):
    def parse(self, text: str) -> Dict[str, str]:
        """Parse the output of an LLM call."""
        match = re.search(self.regex, text, re.DOTALL)
        if match:
            return {key: match.group(i + 1) for i, key in enumerate(self.output_keys)}
        else:
            if self.default_output_key is None:
                raise ValueError(f"Could not parse output: {text}")
            else:
                return {
                    key: text if key == self.default_output_key else ""
                    for key in self.output_keys
                }


def extract_intentful_turns_unique(dialogues):
    turns = set([])
    for dialogue in dialogues:
        for turn in dialogue['turns']:
            if len(turn['intents']):
                turns.add(turn['utterance'])
    return sorted(turns)


def extract_intentful_turns(dialogues):
    turns = []
    for dialogue in dialogues:
        for turn in dialogue['turns']:
            if turn['theme_label'] is not None:
                turns.append({
                    **turn,
                    **turn['theme_label']
                })
    return turns


def turns_to_id_map(turns):
    result = collections.defaultdict(lambda: [])
    for turn in turns:
        result[turn['utterance']].append(turn['utterance_id'])
    return result


def get_llm(llm_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Automatically distribute the model across GPUs
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.03,
        return_full_text=False
    )

    # Wrap the pipeline in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm
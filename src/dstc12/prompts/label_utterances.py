# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from langchain_core.prompts import PromptTemplate


PROMPT = PromptTemplate.from_template(
'''<task>
You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line.
You will also receive a set of theme labels in <themes> </themes> tags. Read through them carefully and associate each utterance with the corresponding theme label index.

<example>
H:
<utterances>
I want to cancel my account
I never received my order
I want to get some information about your insurance offerings
</utterances>

<themes>
1. book a flight
2. information about insurance
3. return product
4. cancel account
5. request refund
6. open account
</themes>

A:
<theme_indices>4, 0, 2</theme_indices>
</example>

<guidance>
Write output in the following format: <theme_indices>comma separated theme indices for every input utterance</theme_indices>
If no theme matches an utterance, assign it the index 0. If multiple themes match an utterance, assign it the theme you thought of first.
</guidance>
</task>

H:
<utterances>
{utterances}
</utterances>

<themes>
{themes}
</themes>
'''
)
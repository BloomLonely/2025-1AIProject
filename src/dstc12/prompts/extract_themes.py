# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from langchain_core.prompts import PromptTemplate


PROMPT = PromptTemplate.from_template(
'''<task>
You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line. Read through them carefully and cluster them into themes. The themes should be exhaustive and mutually exclusive and should cover the dataset completely.
Output a full set of themes you identified. One utterance can only belong to one theme.

<guidance>
Write your output in the following format:
Unique themes number: n
<theme>theme label 1</theme>
<theme>theme label 2</theme>
...
<theme>theme label n</theme>
</guidance>

H:
<utterances>
{utterances}
</utterances>
'''
)
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from langchain_core.prompts import PromptTemplate


PROMPT = PromptTemplate.from_template(
'''<task>
You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line.
These utterances are part of a call center conversation between a customer and an agent.


Your task is to generate a **concise, action-based theme label** that summarizes what the **customer wants to do**.
The label should be:
- fewer than 5 words
- written as a verb phrase (e.g., "open account", "check balance")
- general yet informative


<guidance>
⚠️ You MUST output in the following format **exactly**.
NO extra explanation. NO markdown. Only these two tags in this exact order:


<theme_label_explanation>your reasoning here</theme_label_explanation>
<theme_label>your concise label here</theme_label>


✅ Example:
<theme_label_explanation>Customer wants to see their recent transactions and review activity</theme_label_explanation>
<theme_label>view transaction history</theme_label>
</guidance>
</task>


<utterances>
{{utterances}}
</utterances>
'''

)
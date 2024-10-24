PROMPT_TO_GENERATE_CHOSEN_RESPONSE = """You are a helpful and harmless assistant. Please answer follow the user's instructions below, and output your answer in the requested format.

User instructions:
{instruction}

Please enclose your answer in <answer> tags. For example: <answer>My answer</answer>
"""

PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE = """Below is a conversation between an user and an AI Assistant.

{instruction}

[The start of Assistant's Answer]
{baseline_response}
[The end of Assistant's Answer]

Please first generate a modified instruction that is highly relevant but not semantically identical to \
the instruction above from the user. Then write a high-quality answer which is a good response to the \
modified instruction but not a good response to the original user question. IMPORTANT: Please strictly \
follow the following format:

User Question Modified
<provide a modified instruction here>

The start of Assistant's answer to the modified instruction
<provide a high-quality response to the modified instruction>
The end of Assistant's answer to the modified instruction"""

PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants \
to the user question displayed below. You should choose the assistant that follows the user's instructions \
and answers the user's question better. Your evaluation should consider factors such as the helpfulness, \
relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by \
comparing the two responses and provide a short explanation. Avoid any position biases and ensure that \
the order in which the responses were presented does not influence your decision. Do not allow the length \
of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective \
as possible. After providing your explanation, output your final verdict by strictly following this format: \
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[[User Question]]
{instruction}

[The Start of Assistant A's Answer]
{responsea}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{responseb}
[The End of Assistant B's Answer]
"""

PROMPT_TO_CATEGORISE_INPUTS = """\
I have an instruction below that I would like you to perform three steps of analysis about the instruction:

<instruction> {instruction} </instruction>

Firstly, categorize the instruction above into one of the following categories:

Coding
Mathematical reasoning
Asking for Advice
Brainstorming
Classification
Closed Question Answering
Creative Writing
Extraction
Inhabiting a Character/Persona
Open Question Answering
Rewriting
Summarization
Knowledge and Reasoning
Humanity, History or Social Studies
Other

Secondly, score the instruction in terms of complexity: how complex you think it is to answer from 1-10 \
(where 10 is a complex question whereby first reasoning or breaking down the question into multiple \
subquestions for example might help improve the answer).

Thirdly, indicate how long you think the response to the instruction should be, either (a) 1 sentence, \
(b) 1-3 sentences, (c) 1 paragraph, (d) 2 paragraphs, or (e) 3 or more paragraphs.
Provide your final response in the following format:
Category: <one of the categories above>
Complexity: <score out of 10>
Length: <choose from (a) to (e)>. DO NOT provide the actual response"""

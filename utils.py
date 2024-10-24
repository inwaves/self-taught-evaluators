import pandas as pd
import numpy as np
from datasets import Dataset
from prompts import (
    PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE,
    PROMPT_TO_GENERATE_CHOSEN_RESPONSE,
    PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION,
)
import re
import torch
from vllm import LLM, SamplingParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANSWER_SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE_SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
JUDGEMENT_SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)


def batch_generate(model: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    """
    Generate responses for a batch of prompts using vLLM.

    Args:
        model (LLM): The vLLM model to use for generation.
        prompts (list[str]): List of prompts to generate responses for.
        sampling_params (SamplingParams): Sampling parameters for generation.

    Returns:
        list[str]: List of generated responses.
    """
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def generate_chosen_responses(model: LLM, user_instructions: list[str]) -> list[str]:
    """
    Generate chosen responses for a batch of user instructions.

    Args:
        model (LLM): The vLLM model to use for generation.
        user_instructions (list[str]): List of user instructions.

    Returns:
        list[str]: List of generated chosen responses.
    """
    assembled_prompts = [PROMPT_TO_GENERATE_CHOSEN_RESPONSE.format(instruction=instruction) for instruction in user_instructions]
    responses = batch_generate(model, assembled_prompts, ANSWER_SAMPLING_PARAMS)
    
    chosen_responses = []
    for response in responses:
        assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        match = re.search(r'<answer>(.*?)(?:</answer>|$)', assistant_response, re.DOTALL)
        if match:
            chosen_response = match.group(1).strip()
        else:
            chosen_response = assistant_response
        chosen_response = re.sub(r'^<answer>|<answer>$', '', chosen_response)
        chosen_response = re.sub(r'^</answer>|</answer>$', '', chosen_response)
        chosen_responses.append(chosen_response.strip())
    
    return chosen_responses


def generate_modified_instructions_and_rejected_responses(
    model: LLM, original_prompts: list[str], chosen_responses: list[str]
) -> tuple[list[str], list[str]]:
    """
    Generate modified instructions and rejected responses for a batch of original prompts and chosen responses.

    Args:
        model (LLM): The vLLM model to use for generation.
        original_prompts (list[str]): List of original prompts containing user instructions.
        chosen_responses (list[str]): List of chosen responses to the original prompts.

    Returns:
        tuple[list[str], list[str]]: A tuple containing lists of modified instructions and rejected responses.
    """
    assembled_prompts = [
        PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE.format(
            instruction=prompt, baseline_response=response
        )
        for prompt, response in zip(original_prompts, chosen_responses)
    ]
    
    responses = batch_generate(model, assembled_prompts, MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE_SAMPLING_PARAMS)
    
    modified_instructions = []
    rejected_responses = []
    for response in responses:
        assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        modified_instruction = extract_modified_instruction(assistant_response)
        rejected_response = extract_rejected_response(assistant_response)
        modified_instructions.append(modified_instruction)
        rejected_responses.append(rejected_response)
    
    return modified_instructions, rejected_responses


def extract_modified_instruction(raw_response: str) -> str:
    """
    Extract the modified instruction from the raw response using regex.

    Args:
        raw_response (str): The raw response containing the modified instruction.

    Returns:
        str: The extracted modified instruction or "NO INSTRUCTION" if not found.
    """
    pattern = r"User Question Modified\s*([\s\S]*?)\s*(?:The start of Assistant's answer|$)"
    match = re.search(pattern, raw_response, re.IGNORECASE)
    
    if not match:
        return "NO INSTRUCTION"
    
    return match.group(1).strip()


def extract_rejected_response(raw_response: str) -> str:
    """
    Extract the rejected response from the raw response using regex.

    Args:
        raw_response (str): The raw response containing the rejected response.

    Returns:
        str: The extracted rejected response or "NO REJECTED ANSWER" if not found.
    """
    start_pattern = r"The start of Assistant's answer(?:\s*to the modified instruction)?\s*([\s\S]*)"
    start_match = re.search(start_pattern, raw_response, re.IGNORECASE)
    
    if not start_match:
        return "NO REJECTED ANSWER"
    
    content = start_match.group(1).strip()
    
    end_pattern = r"([\s\S]*?)\s*The end of Assistant's answer(?:\s*to the modified instruction)?"
    end_match = re.search(end_pattern, content, re.IGNORECASE)
    
    if end_match:
        return end_match.group(1).strip()
    else:
        return content if content else "NO REJECTED ANSWER"


def generate_response_pairs_and_modified_instructions(
    model: LLM, user_instructions: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """
    Generate chosen responses, rejected responses, and modified instructions for a batch of user instructions.

    Args:
        model (LLM): The vLLM model to use for generation.
        user_instructions (list[str]): List of user instructions.

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing lists of chosen responses, rejected responses, and modified instructions.
    """
    chosen_responses = generate_chosen_responses(model, user_instructions)
    modified_instructions, rejected_responses = generate_modified_instructions_and_rejected_responses(
        model, user_instructions, chosen_responses
    )
    return chosen_responses, rejected_responses, modified_instructions


def generate_multiple_judgements(model: LLM, prompts: list[str], num_judgements: int) -> list[list[str]]:
    """
    Generate multiple judgements for a batch of prompts.

    Args:
        model (LLM): The vLLM model to use for generation.
        prompts (list[str]): List of input prompts for generating judgements.
        num_judgements (int): The number of judgements to generate for each prompt.

    Returns:
        list[list[str]]: A list of lists containing generated judgements for each prompt.
    """
    all_prompts = prompts * num_judgements
    all_responses = batch_generate(model, all_prompts, JUDGEMENT_SAMPLING_PARAMS)
    
    judgements = []
    for response in all_responses:
        match = re.search(r"\[\[([AB])\]\]", response)
        if match:
            judgement = match.group(1)
        elif "assistant a is better" in response.lower():
            judgement = "A"
        elif "assistant b is better" in response.lower():
            judgement = "B"
        else:
            judgement = "NOT FOUND"
        judgements.append(judgement)
    
    # Group judgements by prompt, ensuring each sublist contains judgements for a single prompt
    return [judgements[i:i+num_judgements] for i in range(0, len(judgements), num_judgements)]


def rejection_sample_judgements(
    generated_judgements: list[str], ground_truth_judgement: str
) -> list[str]:
    """
    Perform rejection sampling on generated judgements based on a single ground truth judgement.

    This function compares each generated judgement with the ground truth judgement
    and only keeps the generated judgements that agree with the ground truth.

    Args:
        generated_judgements (list[str]): A list of judgements generated by a model.
        ground_truth_judgement (str): The single ground truth judgement to compare against.

    Returns:
        list[str]: A list of generated judgements that agree with the ground truth judgement.
    """

    def do_judgements_agree(
        generated_judgement: str, ground_truth_judgement: str
    ) -> bool:
        return generated_judgement == ground_truth_judgement

    return [
        generated_judgement
        for generated_judgement in generated_judgements
        if do_judgements_agree(generated_judgement, ground_truth_judgement)
    ]


def generate_preference_data(model: LLM, dataset: Dataset, num_judgements: int) -> Dataset:
    """
    Generate preference data based on the provided dataset and model.

    This function generates response pairs, modified instructions, and judgements for each datapoint in the dataset.
    It then performs rejection sampling on the judgements and saves the resulting dataset to disk.

    Args:
        model (LLM): The vLLM model to use for generation.
        dataset (Dataset): The input dataset containing user instructions.
        num_judgements (int): The number of judgements to generate for each datapoint.

    Returns:
        Dataset: The generated preference dataset.
    """
    df = dataset.to_pandas()
    user_instructions = df["instruction"].tolist()

    print(f"Generating response pairs and modified instructions for {len(df)} datapoints...")
    chosen_responses, rejected_responses, modified_instructions = generate_response_pairs_and_modified_instructions(
        model, user_instructions
    )
    df["chosen"] = chosen_responses
    df["rejected"] = rejected_responses
    df["modified_instruction"] = modified_instructions
    print("COMPLETE!")

    df["rejected"] = df["rejected"].replace("NO REJECTED ANSWER", pd.NA)
    df["modified_instruction"] = df["modified_instruction"].replace("NO INSTRUCTION", pd.NA)
    df.dropna(subset=["chosen", "rejected", "modified_instruction"], inplace=True)

    print(f"After removing data points with no valid rejected responses, {len(df)} datapoints remain.")

    df["is_chosen_first"] = np.random.choice([True, False], size=len(df))
    df["whois_chosen"] = df["is_chosen_first"].map({True: "A", False: "B"})

    df["prompt"] = df.apply(
        lambda row: PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION.format(
            instruction=row["instruction"],
            responsea=row["chosen"] if row["is_chosen_first"] else row["rejected"],
            responseb=row["rejected"] if row["is_chosen_first"] else row["chosen"],
        ),
        axis=1,
    )

    print(f"Generating {num_judgements} judgements for each of the {len(df)} datapoints...")
    judgements = generate_multiple_judgements(model, df["prompt"].tolist(), num_judgements)
    df["judgements"] = judgements

    def rejection_sample_judgements_or_nan(generated_judgements: list[str], ground_truth_judgement: str) -> str:
        valid_judgements = rejection_sample_judgements(generated_judgements, ground_truth_judgement)
        return np.random.choice(valid_judgements) if valid_judgements else np.nan

    df["retained_judgement"] = df.apply(
        lambda row: rejection_sample_judgements_or_nan(row["judgements"], row["whois_chosen"]),
        axis=1,
    )

    print("COMPLETE!")

    df.dropna(subset=["retained_judgement"], inplace=True)
    print(f"After removing data points with no valid judgements, {len(df)} datapoints remain.")

    training_dataset = Dataset.from_pandas(df)
    training_dataset.save_to_disk("datasets/training_dataset")

    return training_dataset


def standardise_wildchat_dataset(dataset: Dataset) -> Dataset:
    """
    Standardise the WildChat dataset to the format required for training.
    """
    user_instructions = [datapoint["conversation"][0]["content"] for datapoint in dataset]
    df = pd.DataFrame({"instruction": user_instructions})

    print(f"Removing user instruction duplicates...\nLength of dataset before: {len(df)}")
    df.drop_duplicates(subset=["instruction"], inplace=True)
    print(f"Removing user instruction duplicates... COMPLETE\nLength of dataset after: {len(df)}")
    return Dataset.from_pandas(df)
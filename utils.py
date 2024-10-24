import pandas as pd
import numpy as np
from datasets import Dataset
from prompts import (
    PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE,
    PROMPT_TO_GENERATE_CHOSEN_RESPONSE,
    PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION,
)
import re
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_chosen_response(model: PreTrainedModel, tokeniser: PreTrainedTokenizer, user_instruction: str, max_new_tokens: int) -> str:
    """
    Generate a chosen response based on the user instruction using the provided model and tokeniser.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        user_instruction (str): The user's instruction.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated chosen response.

    Raises:
        ValueError: If the chosen response is not found in the raw response.
    """
    assembled_prompt = PROMPT_TO_GENERATE_CHOSEN_RESPONSE.format(
        instruction=user_instruction,
    )
    # Apply chat template and tokenise
    inputs = tokeniser(tokeniser.apply_chat_template([{"role": "user", "content": assembled_prompt}], tokenize=False), return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokeniser.decode(outputs[0])

    # Extract the answer from the response
    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    match = re.search(r'<answer>(.*?)(?:</answer>|$)', assistant_response, re.DOTALL)
    if match:
        chosen_response = match.group(1).strip()
    else:
        # If no <answer> tag is found, use the entire assistant response
        chosen_response = assistant_response

    # Remove any incomplete <answer> tags at the beginning or end
    chosen_response = re.sub(r'^<answer>|<answer>$', '', chosen_response)
    chosen_response = re.sub(r'^</answer>|</answer>$', '', chosen_response)

    return chosen_response.strip()


def generate_modified_instruction_and_rejected_response(
    model: PreTrainedModel, tokeniser: PreTrainedTokenizer, original_prompt: str, chosen_response: str, max_new_tokens: int
) -> tuple[str, str]:
    """
    Generate a modified instruction and a rejected response based on the original prompt and chosen response.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        original_prompt (str): The original prompt containing user instructions.
        chosen_response (str): The chosen response to the original prompt.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        tuple[str, str]: A tuple containing the modified instruction and the rejected response.
    """
    assembled_prompt = (
        PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE.format(
            instruction=original_prompt,
            baseline_response=chosen_response,
        )
    )

    tokenised_input = tokeniser(tokeniser.apply_chat_template([{"role": "user", "content": assembled_prompt}], tokenize=False), return_tensors="pt").to(DEVICE)
    raw_response = model.generate(**tokenised_input, max_new_tokens=max_new_tokens)

    breakpoint()

    decoded_response = tokeniser.decode(raw_response[0])
    assistant_response = decoded_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    # Extract the modified instruction and rejected response from the raw response
    modified_instruction = extract_modified_instruction(assistant_response)
    rejected_response = extract_rejected_response(assistant_response)

    return modified_instruction, rejected_response


def extract_modified_instruction(raw_response: str) -> str:
    """
    Extract the modified instruction from the raw response using regex.

    Args:
        raw_response (str): The raw response containing the modified instruction.

    Returns:
        str: The extracted modified instruction.

    Raises:
        ValueError: If the modified instruction is not found in the response.
    """
    pattern = r"User Question Modified\s*([\s\S]*?)\s*(?:The start of Assistant's answer|$)"
    match = re.search(pattern, raw_response, re.IGNORECASE)
    
    if not match:
        raise ValueError("Modified instruction not found in the response.")
    
    return match.group(1).strip()


def extract_rejected_response(raw_response: str) -> str:
    """
    Extract the rejected response from the raw response using regex.

    Args:
        raw_response (str): The raw response containing the rejected response.

    Returns:
        str: The extracted rejected response.

    Raises:
        ValueError: If the rejected response is not found in the response.
    """
    pattern = r"The start of Assistant's answer to the modified instruction\s*([\s\S]*?)\s*The end of Assistant's answer to the modified instruction"
    match = re.search(pattern, raw_response, re.IGNORECASE)
    
    if not match:
        raise ValueError("Rejected response not found in the response.")
    
    return match.group(1).strip()


def generate_response_pair_and_modified_instruction(
    model: PreTrainedModel, tokeniser: PreTrainedTokenizer, user_instruction: str, max_new_tokens: int
) -> tuple[str, str, str]:
    """
    Generate a chosen response, rejected response, and modified instruction based on the user instruction.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        user_instruction (str): The user's instruction.

    Returns:
        tuple[str, str, str]: A tuple containing the chosen response, rejected response, and modified instruction.
    """
    chosen = generate_chosen_response(model, tokeniser, user_instruction, max_new_tokens)
    modified_instruction, rejected = generate_modified_instruction_and_rejected_response(
        model, tokeniser, user_instruction, chosen, max_new_tokens
    )
    return chosen, rejected, modified_instruction


def generate_multiple_judgements(model: PreTrainedModel, tokeniser: PreTrainedTokenizer, prompt: str, num_judgements: int, max_new_tokens: int) -> list[str]:
    """
    Generate multiple judgements based on the given prompt using the provided model.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        prompt (str): The input prompt for generating judgements.
        num_judgements (int): The number of judgements to generate.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        list[str]: A list of generated judgements.
    """
    return [generate_judgement(model, tokeniser, prompt, max_new_tokens) for _ in range(num_judgements)]


def generate_judgement(model: PreTrainedModel, tokeniser: PreTrainedTokenizer, prompt: str, max_new_tokens: int) -> str:
    """
    Generate a judgement based on the given prompt using the provided model.

    This function generates a response using the model, then attempts to extract
    a judgement (either 'A' or 'B') from the response. It first looks for an exact
    match using regex, then tries to infer the judgement from the text if no exact
    match is found. If no judgement can be determined, it raises a ValueError.

    Args:
        model (PreTrainedModel): The model used to generate the response.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        prompt (str): The input prompt for generating the judgement.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The extracted judgement, either 'A' or 'B'.

    Raises:
        ValueError: If no valid judgement can be extracted from the response.
    """

    prompt = tokeniser(tokeniser.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False), return_tensors="pt").to(DEVICE)
    raw_response = model.generate(**prompt, max_new_tokens=max_new_tokens)

    # Use regex to find the judgement
    match = re.search(r"\[\[([AB])\]\]", raw_response)

    if match:
        judgement = match.group(1)
    else:
        # If no exact match is found, try to infer the judgement
        lower_response = raw_response.lower()
        if "assistant a is better" in lower_response:
            judgement = "A"
        elif "assistant b is better" in lower_response:
            judgement = "B"
        else:
            # If still no judgement can be inferred, raise an error
            raise ValueError(
                f"Unable to extract a valid judgement from the response: {raw_response[:100]}..."
            )

    # Log the raw response and extracted judgement
    print(f"Raw response: {raw_response[:100]}...")
    print(f"Extracted judgement: {judgement}")

    return judgement


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
        # For now the check is simple, but later it might be more complex.
        return generated_judgement == ground_truth_judgement

    return [
        generated_judgement
        for generated_judgement in generated_judgements
        if do_judgements_agree(generated_judgement, ground_truth_judgement)
    ]


def generate_preference_data(model: PreTrainedModel, tokeniser: PreTrainedTokenizer, dataset: Dataset, num_judgements: int, max_new_tokens: int) -> Dataset:
    """
    Generate preference data based on the provided dataset and model.

    This function generates response pairs, modified instructions, and judgements for each datapoint in the dataset.
    It then performs rejection sampling on the judgements and saves the resulting dataset to disk.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        num_judgements (int): The number of judgements to generate for each datapoint.
        dataset (Dataset): The input dataset containing user instructions.
    """
    # At this stage the dataset only needs to contain some user instructions.
    df = dataset.to_pandas()

    print(f"Generating response pairs and modified instructions for {len(df)} datapoints...")
    df[["chosen", "rejected", "modified_instruction"]] = pd.DataFrame(
        df.apply(
            lambda row: generate_response_pair_and_modified_instruction(
                model, tokeniser, row["instruction"], max_new_tokens,
            ),
            axis=1,
        )
    )
    print(f"COMPLETE!")

    # Randomize the order of chosen and rejected responses and track which is which
    df["is_chosen_first"] = np.random.choice([True, False], size=len(df))
    df["whois_chosen"] = df["is_chosen_first"].map({True: "A", False: "B"})

    # Assemble each datapoint into the prompt format used in the judgement annotation
    df["prompt"] = df.apply(
        lambda row: PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION.format(
            instruction=row["instruction"],
            responsea=row["chosen"] if row["is_chosen_first"] else row["rejected"],
            responseb=row["rejected"] if row["is_chosen_first"] else row["chosen"],
        ),
        axis=1,
    )

    print(f"Generating {num_judgements} judgements for each of the {len(df)} datapoints...")

    # Generate multiple judgements for each datapoint, then perform rejection sampling.
    # Keep just one judgement per datapoint – but option to increase this later.
    df["judgements"] = df["prompt"].apply(
        lambda prompt: generate_multiple_judgements(model, tokeniser, prompt, num_judgements)
    )
    df["retained_judgement"] = df.apply(
        lambda row: (
            np.random.choice(
                rejection_sample_judgements(row["judgements"], row["whois_chosen"])
            )
            if rejection_sample_judgements(row["judgements"], row["whois_chosen"])
            else np.nan
        ),
        axis=1,
    )

    print(f"COMPLETE!")

    # If there were no valid judgements, drop the datapoint entirely.
    df.dropna(subset=["retained_judgement"], inplace=True)

    print(f"After removing data points with no valid judgements, {len(df)} datapoints remain.")

    training_dataset = Dataset.from_pandas(df)
    training_dataset.save_to_disk("datasets/training_dataset")

    return training_dataset


def tokenise_dataset(dataset: Dataset, tokeniser: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenise the input dataset using the provided tokeniser.

    Args:
        dataset (Dataset): The input dataset to tokenise.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenisation.
        max_length (int, optional): The maximum length of the tokenised sequences. Defaults to 512.

    Returns:
        Dataset: The tokenised dataset.
    """
    def tokenise_function(examples):
        return {k: v.to(DEVICE) for k, v in tokeniser(
            examples["instruction"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).items()}

    # Tokenise the dataset
    tokenised_dataset = dataset.map(tokenise_function, batched=True)

    # Remove the original text column to save memory
    tokenised_dataset = tokenised_dataset.remove_columns(["instruction"])

    # Set the format of the dataset to PyTorch tensors
    tokenised_dataset.set_format("torch")

    return tokenised_dataset


def standardise_wildchat_dataset(dataset: Dataset) -> Dataset:
    """
    Standardise the WildChat dataset to the format required for training.
    """
    user_instructions = [datapoint["conversation"][0]["content"] for datapoint in dataset]
    return Dataset.from_pandas(pd.DataFrame({"instruction": user_instructions}))

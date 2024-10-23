import pandas as pd
import numpy as np
from datasets import Dataset
from prompts import (
    PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE,
    PROMPT_TO_GENERATE_CHOSEN_RESPONSE,
    PROMPT_TO_GENERATE_JUDGEMENT_ANNOTATION,
)
import re
from transformers import PreTrainedTokenizer


def generate_chosen_response(model, tokenizer, user_instruction: str) -> str:
    assembled_prompt = PROMPT_TO_GENERATE_CHOSEN_RESPONSE.format(
        instruction=user_instruction,
    )
    # Apply chat template and tokenize
    inputs = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": assembled_prompt}], tokenize=False), return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0])
    # Extract the answer from the response
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        chosen_response = match.group(1).strip()
    else:
        raise ValueError("Chosen response not found in the raw response.")
    return chosen_response


def generate_modified_instruction_and_rejected_response(
    model, original_prompt: str, chosen_response: str
) -> tuple[str, str]:
    """Given an original prompt x_i containing user instructions,
    this function first generates a modified instruction x'_i, and then
    generates a response y^l_i which is designed to be worse
    than y^w_i, the chosen response to the original prompt x_i.
    args:
        model (BaseModel): the model to use for generation
        original_prompt (str): the original prompt x_i
        chosen_response (str): the chosen response y^w_i
    return:
        modified_instruction (str): the modified instruction x'_i
        rejected_response (str): the rejected response y^l_i
    """

    assembled_prompt = (
        PROMPT_TO_GENERATE_MODIFIED_INSTRUCTION_AND_REJECTED_RESPONSE.format(
            instruction=original_prompt,
            baseline_response=chosen_response,
        )
    )
    raw_response = model.generate(assembled_prompt)

    # Extract the modified instruction and rejected response from the raw response
    modified_instruction = extract_modified_instruction(raw_response)
    rejected_response = extract_rejected_response(raw_response)

    return modified_instruction, rejected_response


def extract_modified_instruction(raw_response: str) -> str:
    """Extract the modified instruction from the raw response using regex."""
    pattern = r"User Question Modified\s*([\s\S]*?)\s*(?:The start of Assistant's answer|$)"
    match = re.search(pattern, raw_response, re.IGNORECASE)
    
    if not match:
        raise ValueError("Modified instruction not found in the response.")
    
    return match.group(1).strip()


def extract_rejected_response(raw_response: str) -> str:
    """Extract the rejected response from the raw response using regex."""
    pattern = r"The start of Assistant's answer to the modified instruction\s*([\s\S]*?)\s*The end of Assistant's answer to the modified instruction"
    match = re.search(pattern, raw_response, re.IGNORECASE)
    
    if not match:
        raise ValueError("Rejected response not found in the response.")
    
    return match.group(1).strip()


def generate_response_pair_and_modified_instruction(
    model, user_instruction: str
) -> tuple[str, str, str]:
    chosen = generate_chosen_response(model, user_instruction)
    modified_instruction, rejected = generate_modified_instruction_and_rejected_response(
        model, user_instruction, chosen
    )
    return chosen, rejected, modified_instruction


def generate_multiple_judgements(model, prompt, num_judgements):
    return [generate_judgement(model, prompt) for _ in range(num_judgements)]


def generate_judgement(model, prompt: str) -> str:
    """
    Generate a judgement based on the given prompt using the provided model.

    This function generates a response using the model, then attempts to extract
    a judgement (either 'A' or 'B') from the response. It first looks for an exact
    match using regex, then tries to infer the judgement from the text if no exact
    match is found. If no judgement can be determined, it raises a ValueError.

    Args:
        model: The model used to generate the response.
        prompt (str): The input prompt for generating the judgement.

    Returns:
        str: The extracted judgement, either 'A' or 'B'.

    Raises:
        ValueError: If no valid judgement can be extracted from the response.
    """
    raw_response = model.generate(prompt)

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


def generate_preference_data(model, num_judgements: int, dataset: Dataset):
    # At this stage the dataset only needs to contain some user instructions.
    df = dataset.to_pandas()

    print(f"Generating response pairs and modified instructions for {len(df)} datapoints...")
    df[["chosen", "rejected", "modified_instruction"]] = pd.DataFrame(
        df.apply(
            lambda row: generate_response_pair_and_modified_instruction(
                model, row["instruction"]
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
        lambda prompt: generate_multiple_judgements(model, prompt, num_judgements)
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


def tokenise_dataset(dataset: Dataset, tokeniser: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenize the input dataset using the provided tokeniser.

    Args:
        dataset (Dataset): The input dataset to tokenise.
        tokeniser (PreTrainedTokenizer): The tokeniser to use for tokenization.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 512.

    Returns:
        Dataset: The tokenized dataset.
    """
    def tokenise_function(examples):
        return tokeniser(
            examples["instruction"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenise_function, batched=True)

    # Remove the original text column to save memory
    tokenized_dataset = tokenized_dataset.remove_columns(["instruction"])

    # Set the format of the dataset to PyTorch tensors
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

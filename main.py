import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_preference_data, standardise_wildchat_dataset

DATASET = "allenai/WildChat-1M"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EXAMPLES = 10
NUM_JUDGEMENTS = 1
MAX_NEW_TOKENS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = load_dataset(DATASET, split="train").take(NUM_EXAMPLES)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"Loaded model & tokeniser: {MODEL}")
    print(f"Using device: {DEVICE}")
    print(f"Loaded dataset: {DATASET}")

    dataset = standardise_wildchat_dataset(dataset)
    print(f"Standardised dataset: {DATASET}")

    preference_dataset = generate_preference_data(model, tokenizer, dataset, NUM_JUDGEMENTS, MAX_NEW_TOKENS)
    print(f"Generated preference dataset: {preference_dataset}")

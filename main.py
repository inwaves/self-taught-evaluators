from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_preference_data, standardise_wildchat_dataset

DATASET = "allenai/WildChat-1M"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EXAMPLES = 10
NUM_JUDGEMENTS = 1

if __name__ == "__main__":
    dataset = load_dataset(DATASET, split="train").take(NUM_EXAMPLES)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = standardise_wildchat_dataset(dataset)
    generate_preference_data(model, tokenizer, NUM_JUDGEMENTS, dataset)

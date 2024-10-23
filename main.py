from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_preference_data

DATASET = ""
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

if __name__ == "__main__":
    dataset = load_dataset(DATASET)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Pass the tokenizer to generate_preference_data instead of tokenizing the whole dataset
    generate_preference_data(model, tokenizer, 1, dataset)

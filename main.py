from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_preference_data

DATASET = "allenai/WildChat-1M"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EXAMPLES = 10
NUM_JUDGEMENTS = 1

if __name__ == "__main__":
    dataset = load_dataset(DATASET).take(NUM_EXAMPLES)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    generate_preference_data(model, tokenizer, NUM_JUDGEMENTS, dataset)

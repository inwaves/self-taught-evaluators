import time
import torch
from datasets import load_dataset
from utils import generate_preference_data, standardise_wildchat_dataset
from vllm import LLM

DATASET = "allenai/WildChat-1M"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_EXAMPLES = 10
NUM_JUDGEMENTS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = load_dataset(DATASET, split="train").take(NUM_EXAMPLES)

    model = LLM(MODEL, tensor_parallel_size=2)
    print(f"Loaded model: {MODEL}")
    print(f"Using device: {DEVICE}")
    print(f"Loaded dataset: {DATASET}")

    dataset = standardise_wildchat_dataset(dataset)
    print(f"Standardised dataset: {DATASET}")

    start_time = time.perf_counter()
    # preference_dataset = generate_preference_data(model, dataset, NUM_JUDGEMENTS)
    end_time = time.perf_counter()
    # print(f"Generated dataset containing {len(preference_dataset)} judgements in {end_time - start_time:.2f} seconds")

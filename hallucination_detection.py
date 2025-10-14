# ==============================================================================
#  Google Colab - Hallucination Detection with Effective Rank (FIXED)
# ==============================================================================
# INSTRUCTIONS:
# 1. Open this in Google Colab (https://colab.research.google.com).
# 2. Go to the menu: `Runtime` -> `Change runtime type`.
# 3. Select `T4 GPU` as the Hardware accelerator.
# 4. Paste this entire script into a single cell and run it.
# ==============================================================================

# --- 1. Installation ---
# FIX #1: Added the --upgrade flag to ensure we get the latest library versions,
# which is crucial for compatibility with new models like Phi-3.
print("--- Installing and upgrading required libraries ---")
!pip install --upgrade -q transformers torch accelerate bitsandbytes

# --- 2. Imports and Helper Functions ---
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress a common warning from the transformers library
warnings.filterwarnings("ignore", message="The linked framework is not supported properly.")

def calculate_effective_rank(matrix):
    """Calculates the effective rank of a matrix based on its singular values."""
    matrix = matrix.astype(np.float64)
    try:
        S = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        print("SVD computation failed.")
        return 0.0

    S_squared = S**2
    S_sum_squared = np.sum(S_squared)
    if S_sum_squared == 0:
        return 1.0

    normalized_S_squared = S_squared / S_sum_squared
    positive_mask = normalized_S_squared > 0
    entropy = -np.sum(
        normalized_S_squared[positive_mask]
        * np.log2(normalized_S_squared[positive_mask])
    )
    effective_rank = float(np.power(2.0, entropy))
    return effective_rank

def get_last_hidden_state(model, input_ids):
    """Runs a forward pass to grab the hidden state of the final token."""
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )

    last_hidden = outputs.hidden_states[-1][:, -1, :]
    return last_hidden

# --- 3. Load the Large Language Model ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
print(f"\n--- Loading model: {MODEL_NAME} ---")
print("This will download ~7.5GB of data. Please be patient...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Phi-3 tokenizer doesn't have a default pad token, so we set it to the eos token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n--- Model loaded successfully onto GPU! ---")

# --- 4. Run the Analysis ---
factual_prompt = "What is the chemical formula for water?"
hallucination_prompt = "Explain the role of the acoustic dampeners in the design of the Great Pyramid of Giza."
NUM_SAMPLES = 30

# --- Process Factual Prompt ---
print("\n--- Analyzing Factual Prompt ---")
print(f"Prompt: '{factual_prompt}'")
collected_vectors_factual = []

for i in range(NUM_SAMPLES):
    messages = [{"role": "user", "content": factual_prompt}]

    # FIX #2: Explicitly create and pass the attention_mask.
    # First, apply the template to get the prompt string.
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Then, tokenize the string to get both input_ids and the attention_mask.
    inputs = tokenizer(prompt_string, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        # Pass the entire `inputs` dictionary using the ** operator.
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            use_cache=False,  # Fix for DynamicCache AttributeError
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # Extract hidden state of the newly generated token via a forward pass
    vector = get_last_hidden_state(model, outputs.sequences)
    collected_vectors_factual.append(vector.detach().float().cpu().numpy().squeeze(0))

    # Decode only the newly generated tokens
    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs.sequences[0, input_length:], skip_special_tokens=True)
    print(f"  Run {i+1:2d}: {generated_text.strip()}")

matrix_factual = np.array(collected_vectors_factual)
rank_factual = calculate_effective_rank(matrix_factual)
print(f"\nMatrix of hidden states has shape: {matrix_factual.shape}")
print(f"--> Effective Rank for Factual Prompt: {rank_factual:.4f} (Low score = High Confidence)")


# --- Process Hallucination Prompt ---
print("\n--- Analyzing Hallucination Prompt ---")
print(f"Prompt: '{hallucination_prompt}'")
collected_vectors_hallucination = []

for i in range(NUM_SAMPLES):
    messages = [{"role": "user", "content": hallucination_prompt}]

    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_string, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            temperature=0.9,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            use_cache=False,  # Fix for DynamicCache AttributeError
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # Extract hidden state of the newly generated token via a forward pass
    vector = get_last_hidden_state(model, outputs.sequences)
    collected_vectors_hallucination.append(vector.detach().float().cpu().numpy().squeeze(0))

    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs.sequences[0, input_length:], skip_special_tokens=True)
    print(f"  Run {i+1:2d}: {generated_text.strip()}")

matrix_hallucination = np.array(collected_vectors_hallucination)
rank_hallucination = calculate_effective_rank(matrix_hallucination)
print(f"\nMatrix of hidden states has shape: {matrix_hallucination.shape}")
print(f"--> Effective Rank for Hallucination Prompt: {rank_hallucination:.4f} (High score = Low Confidence)")

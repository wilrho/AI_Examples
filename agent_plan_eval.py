# ==============================================================================
#  Google Colab - Agent Planning Evaluation
# ==============================================================================
# INSTRUCTIONS:
# 1. Open this in Google Colab (https://colab.research.google.com).
# 2. Go to the menu: `Runtime` -> `Change runtime type`.
# 3. Select `T4 GPU` as the Hardware accelerator.
# 4. Paste this entire script into a single cell and run it.
# ==============================================================================
# ==============================================================================
# Step 1: Install necessary libraries
# ==============================================================================
print("Installing required packages... This may take a few minutes.")
!pip install transformers torch accelerate bitsandbytes --quiet

# ==============================================================================
# Step 2: Setup models and helper functions
# ==============================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import textwrap

print("\nSetup complete. Preparing models and functions...")

# Define the models we will compare (NO LOGIN REQUIRED)
MODEL_SMART = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_LESS_CAPABLE = "HuggingFaceH4/zephyr-7b-beta" # CORRECTED MODEL ID

# --- PROMPT DEFINITION ---
# This prompt includes a few-shot example to enforce the correct JSON structure.
SYSTEM_PROMPT = """
You are an expert planning agent. Your task is to break down a user's goal into a logical sequence of tool calls in JSON format.

You have access to the following tools:
- search_flights(origin: string, destination: string, date: string)
- search_hotels(city: string, near_landmark: string, required_amenities: list[string])
- find_points_of_interest(latitude: float, longitude: float, poi_type: string)
- book_trip(flight_details: object, hotel_details: object)

CRITICAL INSTRUCTIONS:
1.  The final output MUST be a valid JSON array of objects, where each object has a "tool_name" and a "params" key.
2.  Recognize dependencies. For unknown parameters, use a placeholder string like "<RESULT_OF_STEP_2>".
3.  The user said "Don't book anything yet", so you MUST NOT use the 'book_trip' tool.
4.  Today is October 17th, 2025 (a Friday). "Next Friday" is October 24th, 2025.
5.  Respond ONLY with the JSON. Do not add any other text, explanations, or markdown formatting.

EXAMPLE:
User Goal: "Find a flight to London and a hotel near the Tower of London."
Your Response:
[
  {
    "tool_name": "search_flights",
    "params": {
      "origin": "current_location",
      "destination": "London",
      "date": "today"
    }
  },
  {
    "tool_name": "search_hotels",
    "params": {
      "city": "London",
      "near_landmark": "Tower of London",
      "required_amenities": []
    }
  }
]
"""

USER_GOAL = "My team is having an offsite in Melbourne. Find me a flight from Sydney for next Friday. Also, find a hotel near the convention centre that has a 'meeting room'. Oh, and my boss loves Italian food, so see if there's a good restaurant nearby. Don't book anything yet, just give me the plan."

# --- Helper Functions ---
def load_model_pipeline(model_id):
    """Loads a quantized model and tokenizer."""
    print(f"\nLoading model: {model_id}. This may take several minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"Finished loading {model_id}.")
    return pipe

def generate_plan(pipe, system_prompt, user_goal):
    """Generates a plan using the provided model pipeline and system prompt."""
    messages = [{"role": "user", "content": f"{system_prompt}\n\nUser Goal: {user_goal}"}]
    # Note: Zephyr uses a slightly different chat template format
    if "zephyr" in pipe.tokenizer.name_or_path.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal}
        ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(prompt, max_new_tokens=512, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id)
    generated_text = outputs[0]['generated_text'][len(prompt):]
    
    try:
        json_start = generated_text.find('[')
        json_end = generated_text.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            return json.loads(json_str)
        return {"error": "Valid JSON array not found.", "raw_output": generated_text}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON.", "raw_output": generated_text}

def analyze_plan(plan):
    """Evaluates the generated plan and prints the analysis."""
    if not isinstance(plan, list) or not plan:
        print(f"  - 🔴 CRITICAL FAIL: The model did not produce a valid plan list.")
        raw_output = plan.get('raw_output', 'N/A') if isinstance(plan, dict) else str(plan)
        wrapped_output = textwrap.fill(raw_output, width=80, initial_indent="    ", subsequent_indent="    ")
        print(f"    Raw Output:\n{wrapped_output}")
        return

    issues = 0

    # Test 0: Basic Formatting
    if all(isinstance(step, dict) and 'tool_name' in step and 'params' in step for step in plan):
         print("  - 🟢 PASS: Adhered to the required JSON schema. (Evaluates: Tool Use)")
    else:
        print("  - 🔴 FAIL: Did not adhere to the required JSON schema (list of dicts with 'tool_name' and 'params'). (Evaluates: Tool Use)")
        issues += 1
    
    # Test 1: Dependency Planning (Orchestration)
    hotel_index = next((i for i, step in enumerate(plan) if isinstance(step, dict) and step.get('tool_name') == 'search_hotels'), -1)
    poi_index = next((i for i, step in enumerate(plan) if isinstance(step, dict) and step.get('tool_name') == 'find_points_of_interest'), -1)
    if hotel_index != -1 and poi_index != -1 and hotel_index < poi_index:
        print("  - 🟢 PASS: Correctly planned to find hotel before nearby restaurants. (Evaluates: Planning)")
    else:
        print("  - 🔴 FAIL: Incorrectly ordered the search for hotel and restaurants, or missed a step. (Evaluates: Planning)")
        issues += 1

    # Test 2: Parameter Extraction
    hotel_step = next((step for step in plan if isinstance(step, dict) and step.get('tool_name') == 'search_hotels'), None)
    amenities = hotel_step.get('params', {}).get('required_amenities', []) if hotel_step else []
    # CORRECTED CHECK: Now accepts space or underscore.
    if 'meeting room' in amenities or 'meeting_room' in amenities:
        print("  - 🟢 PASS: Correctly extracted 'meeting room' as a required amenity. (Evaluates: Tool Use)")
    else:
        print("  - 🔴 FAIL: Missed the 'meeting room' constraint. (Evaluates: Tool Use)")
        issues += 1
        
    # Test 3: Negative Constraint
    book_step_found = any(isinstance(step, dict) and step.get('tool_name') == 'book_trip' for step in plan)
    if not book_step_found:
        print("  - 🟢 PASS: Correctly followed the instruction 'Don't book anything'. (Evaluates: Tool Use)")
    else:
        print("  - 🔴 FAIL: Ignored the negative constraint and planned to book the trip. (Evaluates: Tool Use)")
        issues += 1

    print("\n  CONCLUSION:")
    if issues == 0:
        print("    This agent demonstrates ROBUST planning and tool use capabilities.")
    else:
        print(f"    This agent shows {issues} CLEAR FAILURE(S), highlighting the critical need for evaluation.")


# ==============================================================================
# Step 3: Run the Evaluation
# ==============================================================================

def run_evaluation(model_id, model_name):
    """Loads a model and runs the full plan generation and analysis."""
    pipe = load_model_pipeline(model_id)
    print("\n" + "="*60)
    print(f"  EVALUATING: {model_name}")
    print("="*60)
    
    plan = generate_plan(pipe, SYSTEM_PROMPT, USER_GOAL)
    
    print("\nGenerated Plan:")
    print(json.dumps(plan, indent=2))
    print("\nEvaluation Analysis:")
    analyze_plan(plan)
    
    # Clean up to free memory
    del pipe
    del plan
    torch.cuda.empty_cache()

# --- SMART MODEL EVALUATION ---
run_evaluation(MODEL_SMART, "SMART MODEL (Mistral-7B)")

# --- LESS CAPABLE MODEL EVALUATION ---
run_evaluation(MODEL_LESS_CAPABLE, "LESS CAPABLE MODEL (Zephyr-7B)")

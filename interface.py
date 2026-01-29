# This is for command-line interface to demonstrate instruction tuning using LoRA with GPT-2 model.
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

MODEL_NAME = "distilgpt2"
LORA_PATH = "./lora_adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def generate(instruction):
    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interface.py \"Your instruction here\"")
        sys.exit(1)

    instruction = sys.argv[1]
    response = generate(instruction)
    print("\n=== Model Response ===\n")
    print(response)

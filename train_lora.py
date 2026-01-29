# Train a GPT-2 model with LoRA adapters on the Alpaca dataset.
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

# --------------------
# Config
# --------------------
MODEL_NAME = "distilgpt2"
DATASET_NAME = "yahma/alpaca-cleaned"
OUTPUT_DIR = "./lora_adapter"
MAX_LENGTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load dataset
# --------------------
dataset = load_dataset(DATASET_NAME, split="train[:10000]")

# --------------------
# Tokenizer
# --------------------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}
"""
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# --------------------
# Model + LoRA
# --------------------
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
base_model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(DEVICE)

# --------------------
# Training arguments
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# --------------------
# Save LoRA adapters
# --------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA training complete. Adapters saved.")

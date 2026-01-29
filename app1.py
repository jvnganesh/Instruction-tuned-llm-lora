# This is for streamlit app to demonstrate instruction tuning using LoRA with GPT-2 model.
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "distilgpt2"
LORA_PATH = "./lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="LoRA Instruction-Tuned GPT-2", layout="centered")

st.title("🧠 Instruction-Tuned GPT-2 (LoRA)")
st.write("Efficient instruction tuning using **LoRA (PEFT)**")

instruction = st.text_area(
    "Instruction",
    placeholder="Explain gradient descent in simple terms",
    height=120
)

max_length = st.slider("Max Length", 50, 300, 150)
temperature = st.slider("Temperature", 0.1, 1.5, 0.5)

if st.button("Generate"):
    if instruction.strip() == "":
        st.warning("Please enter an instruction.")
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Model Response")
        st.write(response)

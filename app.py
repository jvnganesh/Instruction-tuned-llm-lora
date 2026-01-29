# This is for gradio app to demonstrate instruction tuning using LoRA with GPT-2 model.
import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
model.eval()

def generate_response(instruction, max_length, temperature):
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks(title="Instruction-Tuned GPT-2 (LoRA)") as demo:
    gr.Markdown("## 🧠 Instruction-Tuned GPT-2 (LoRA)")
    gr.Markdown("Efficient instruction tuning using LoRA + Alpaca dataset")

    instruction = gr.Textbox(
        label="Instruction",
        placeholder="Explain gradient descent in simple terms",
        lines=4
    )

    max_length = gr.Slider(50, 300, value=150, label="Max Length")
    temperature = gr.Slider(0.1, 1.5, value=0.5, label="Temperature")

    output = gr.Textbox(label="Model Response", lines=10)

    gr.Button("Generate").click(
        fn=generate_response,
        inputs=[instruction, max_length, temperature],
        outputs=output
    )

demo.launch(share=True)

# 🧠 Instruction-Tuned LLM using LoRA (DistilGPT-2)

An end-to-end project demonstrating **efficient fine-tuning and deployment of a Large Language Model (LLM)** using **Low-Rank Adaptation (LoRA)** and an **interactive Gradio interface**.

> 🎯 **Focus**: LLM systems engineering — training, optimization, evaluation, and deployment  
> ❌ **Not** a ChatGPT replacement

---

## 📌 Project Overview

Fine-tuning large language models is expensive and resource-intensive.  
This project shows how **Parameter-Efficient Fine-Tuning (PEFT)** can adapt a pretrained model to follow instructions **without retraining all parameters**.

### 🔹 What this project does
- Uses a **pretrained DistilGPT-2** language model
- Performs **instruction tuning** using the Alpaca dataset
- Applies **LoRA (PEFT)** to drastically reduce training cost
- Deploys the model via a **Gradio web interface**
- Highlights **real-world trade-offs of small LLMs**

---

## 🧩 Architecture Overview

User Instruction
↓
Prompt Formatting (Instruction → Response)
↓
Base Model (DistilGPT-2, frozen)
↓
LoRA Adapters (trainable)
↓
Text Generation (Inference)
↓
Gradio Web Interface


---

## 🛠 Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| Model | DistilGPT-2 (82M parameters) |
| Fine-Tuning | Hugging Face Transformers |
| Efficiency | PEFT (LoRA) |
| Dataset | Alpaca (Instruction–Response) |
| Training | PyTorch, Hugging Face Trainer |
| Deployment | Gradio |
| Environment | Google Colab (GPU) |

---

## 📚 Dataset

### Alpaca Instruction Dataset
- **Format**: Instruction → Response  
- **Samples used**: ~10,000  
- **Purpose**: Teach the model *how to respond to instructions*, not just generate text

**Example**
Instruction: Explain gradient descent.
Response: Gradient descent is an optimization algorithm...


---

## 🔍 Why DistilGPT-2?

- Lightweight and fast
- Ideal for **educational and demo purposes**
- Clearly demonstrates **limitations of small LLMs**
- Excellent choice for showcasing **LoRA efficiency**

⚠️ *DistilGPT-2 is not designed for deep reasoning or factual accuracy.*

---

## ⚡ Why LoRA (Low-Rank Adaptation)?

Instead of fine-tuning **all 82M parameters**, LoRA:

- Freezes the base model
- Trains only small **low-rank adapter matrices** in attention layers

### ✅ Benefits
- 🚀 **10–20× faster training**
- 💾 **~98% fewer trainable parameters**
- 🔥 Industry-standard approach for LLM fine-tuning

**Example Training Stats**
Trainable parameters: ~1.6M
Total parameters: ~82M
Trainable %: ~1.9%


---

## 🧪 Training Details

| Parameter | Value |
|---------|------|
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Max Sequence Length | 256 |
| Precision | FP16 |
| Fine-Tuning Method | LoRA (PEFT) |

Training was performed on **GPU (Google Colab)**.

---

## 📊 Evaluation

The model was evaluated using:
- **Perplexity** (language modeling quality)
- **Qualitative comparison** with base DistilGPT-2
- Manual inspection of instruction-following behavior

> ✅ Priority was **engineering correctness**, not output perfection.

---

## 🌐 Gradio Web Demo

The LoRA-tuned model is deployed using **Gradio**, enabling:
- Live instruction input
- Adjustable decoding parameters
- Real-time text generation
- Public shareable link (Colab compatible)

### Demo Features
- Instruction text box
- Temperature & max-length controls
- GPU / CPU auto-detection

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the model (optional)
python train_lora.py
3️⃣ Run CLI inference
python interface.py "Explain gradient descent in simple terms"
4️⃣ Launch Gradio app
python app.py
📁 Repository Structure
instruction-tuned-llm-lora/
│
├── train_lora.py        # LoRA fine-tuning script
├── interface.py         # CLI inference
├── app.py               # Gradio web app
├── app1.py              # Streamlit web app
├── requirements.txt
├── README.md
│
├── lora_adapter/        # LoRA adapter weights only
│   ├── adapter_config.json
│   └── adapter_model.safetensors

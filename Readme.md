# 🦙 Fine-Tuning LLaMA 2 with QLoRA & PEFT

This project demonstrates how to fine-tune the LLaMA 2 model using **QLoRA** (Quantized Low-Rank Adaptation) and **PEFT** (Parameter-Efficient Fine-Tuning). It enables training large language models on consumer-grade hardware efficiently.

---

## 🚀 Overview

We fine-tune the `meta-llama/Llama-2-7b-hf` model using:
- **4-bit quantization** (QLoRA)
- **LoRA adapters** for efficient training
- A simple dataset of English quotes

The result is a lightweight fine-tuned model that can be used for text generation.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install transformers datasets peft bitsandbytes accelerate trl
```

---

## 📁 Project Structure

- `Fine_tune_Llama_2.ipynb` – Main notebook for training and inference  
- `README.md` – Project overview  
- `lora-llama2/` – (Created after training) Contains fine-tuned LoRA weights  

---

## 🧠 Key Concepts

- **PEFT**: Fine-tunes only a small number of parameters to save resources  
- **LoRA**: Injects trainable low-rank matrices into transformer layers  
- **QLoRA**: Combines LoRA with 4-bit quantized models to reduce memory usage  

---

## 📊 Dataset

We use the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset from Hugging Face for demonstration.

---

## 🔧 Steps

1. Load the LLaMA 2 model in 4-bit  
2. Tokenize the dataset  
3. Apply LoRA using PEFT  
4. Train using Hugging Face `Trainer`  
5. Save LoRA weights  
6. Reload and run inference  

---

## 🧪 Example Inference

```python
input_text = "Success is not final,"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = lora_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💾 Output

After training, only the **LoRA adapter weights** are saved. You can reuse them by loading the base model and applying the adapter:

```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(...)
lora_model = PeftModel.from_pretrained(base_model, "lora-llama2")
```

---

## 📘 References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [PEFT Library](https://github.com/huggingface/peft)  
- [LLaMA 2](https://huggingface.co/meta-llama)  
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)  

---

## 🛠️ Author

Built with ❤️ using Hugging Face and PEFT by Bhuvan Shingade.

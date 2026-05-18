# Medical AI Assistant - LLM Fine-tuning with QLoRA

This project fine-tunes a Llama 3.2 1B language model on medical data using QLoRA (Quantized Low-Rank Adaptation) technique in Google Colab. The final model answers only medical questions and refuses all non-medical queries.

---

## How It Works - Step by Step

---

### Step 1: GPU Check

```python
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")
else:
    print("No GPU found! Go to Runtime > Change Runtime Type > T4 GPU")
```

Before doing anything, we check whether a GPU is available in the Colab environment. `torch.cuda.is_available()` returns True if CUDA (GPU support) is active. We then print the GPU name and total VRAM (Video RAM) in gigabytes. Fine-tuning requires a GPU — without one, training would take hours or days instead of minutes. If no GPU is found, you need to switch the runtime in Google Colab.

---

### Step 2: Hugging Face Login

```python
from huggingface_hub import login
from google.colab import userdata

try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print("Login successful!")
except:
    login()
```

We log in to Hugging Face to access gated models like Llama 3.2. The token (`HF_TOKEN`) is stored as a secret in Colab's userdata (under the key icon in the sidebar) so it is not hardcoded in the script. If the secret is not found, a manual popup appears asking for the token. You must request access to the Llama model on the Hugging Face model page before this works.

---

### Step 3: Clean Install of All Libraries

```bash
pip uninstall -y transformers peft accelerate bitsandbytes trl datasets \
    huggingface_hub triton torch torchvision torchaudio

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 \
    datasets==3.1.0 trl==0.12.1 huggingface_hub==0.26.2 \
    triton==3.1.0 bitsandbytes==0.44.1
```

We first uninstall any existing versions of all relevant libraries to avoid version conflicts. Then we install specific, tested versions of each library. The QLoRA stack requires these libraries to work together in a compatible way:

- `torch` — the core deep learning framework (PyTorch)
- `transformers` — loads pre-trained models and tokenizers from Hugging Face
- `peft` — provides LoRA and other parameter-efficient fine-tuning methods
- `accelerate` — handles multi-GPU and mixed-precision training
- `bitsandbytes` — enables 4-bit quantization to reduce memory usage
- `trl` — contains Trainer utilities for language model fine-tuning
- `datasets` — loads and processes training data
- `triton` — GPU kernel library required by bitsandbytes

---

### Step 4: Upgrade bitsandbytes and Restart

```python
pip install bitsandbytes==0.46.1

import os
os.kill(os.getpid(), 9)
```

After the initial install, we upgrade bitsandbytes to a newer version that has better 4-bit quantization support. Then `os.kill(os.getpid(), 9)` forcefully kills the current Python process, which triggers a Colab runtime restart. This is required so the newly installed libraries are properly loaded into memory. Colab will automatically reconnect and you continue from the next cell.

---

### Step 5: Verify All Libraries Are Loaded Correctly

```python
import torch
import transformers
import peft
import bitsandbytes as bnb

print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"BNB: {bnb.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

After restart, we import all key libraries and print their version numbers. This is a sanity check to confirm everything installed correctly and the GPU is still accessible. If any import fails here, there is a dependency issue that must be resolved before proceeding.

---

### Step 6: Create the Medical Training Dataset

```python
medical_data = [
    {
        "instruction": "Patient has fever of 102F...",
        "output": "Likely diagnoses: ..."
    },
    ...
]

with open('medical_data.json', 'w') as f:
    json.dump(medical_data, f, indent=2)
```

We define a list of 12 medical question-answer pairs covering topics like fever, diabetes, heart attack, blood pressure, appendicitis, pediatric emergencies, and more. Each entry has an `instruction` (the medical question) and an `output` (the expert medical answer). This data is then saved to a `medical_data.json` file. This is the dataset the model will learn from.

---

### Step 7: Load the Base Model with 4-bit Quantization

```python
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

This is the core loading step. `BitsAndBytesConfig` sets up 4-bit quantization:

- `load_in_4bit=True` — loads model weights using only 4 bits per parameter instead of 16 or 32, reducing memory usage by ~4x
- `bnb_4bit_use_double_quant=True` — applies a second round of quantization on the quantization constants themselves, saving a small amount of extra memory
- `bnb_4bit_quant_type="nf4"` — uses NormalFloat4 format, which is mathematically optimal for normally distributed weights
- `bnb_4bit_compute_dtype=torch.bfloat16` — performs actual computations in bfloat16 for speed, even though weights are stored in 4-bit

The tokenizer converts text into numbers the model understands. We set `pad_token = eos_token` because Llama does not have a dedicated padding token. `gradient_checkpointing_enable()` reduces memory during training by recomputing some intermediate values instead of storing them. `prepare_model_for_kbit_training` makes the quantized model trainable.

---

### Step 8: Apply LoRA Adapters

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

LoRA (Low-Rank Adaptation) is the key technique that makes fine-tuning practical. Instead of updating all model parameters (which requires enormous memory), LoRA freezes the original weights and adds small trainable matrices (adapters) alongside specific layers. Only about 1-2% of parameters are trained, drastically reducing GPU memory requirements.

- `r=16` — the rank of the LoRA matrices; higher rank = more capacity but more memory
- `lora_alpha=32` — scaling factor for the LoRA updates; typically set to 2x the rank
- `target_modules` — which layers to apply LoRA to; these are the attention and feed-forward projection layers inside each transformer block
- `lora_dropout=0.05` — randomly drops 5% of LoRA connections during training to prevent overfitting
- `task_type="CAUSAL_LM"` — tells PEFT this is a causal language modeling task (predict next token)

---

### Step 9: Format Data into Prompt Templates

```python
SYSTEM_PROMPT = """You are a specialized Medical AI Assistant trained exclusively on medical knowledge.
Only answer medical and health-related questions..."""

def create_prompt(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{sample['instruction']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{sample['output']}<|eot_id|><|end_of_text|>"""
    return {"text": prompt}
```

The raw question-answer pairs are wrapped in the Llama 3 chat template format using special tokens like `<|begin_of_text|>`, `<|start_header_id|>`, and `<|eot_id|>`. This is the exact format Llama 3 was pre-trained with, so using this structure helps the model understand the conversation roles (system, user, assistant). The system prompt is included in every training example to teach the model its identity and limitations.

The dataset is then split: 90% for training and 10% for evaluation.

---

### Step 10: Tokenize the Dataset

```python
def tokenize(sample):
    return tokenizer(
        sample["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

train_tokenized = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
eval_tokenized = eval_data.map(tokenize, batched=True, remove_columns=eval_data.column_names)
```

The model does not understand text — it only works with numbers. Tokenization converts each prompt string into a list of token IDs. `max_length=512` means each prompt is capped at 512 tokens. Shorter prompts are padded with the eos token to reach 512. The original text column is removed since the model only needs the token IDs during training.

---

### Step 11: Configure and Run Training

```python
training_args = transformers.TrainingArguments(
    output_dir="./medical_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)
```

These are the hyperparameters that control how training runs:

- `num_train_epochs=3` — the model sees the entire dataset 3 times
- `per_device_train_batch_size=2` — processes 2 examples at a time per GPU
- `gradient_accumulation_steps=4` — accumulates gradients over 4 batches before updating weights, effectively simulating a batch size of 8 while using less memory
- `learning_rate=2e-4` — how much to adjust weights each step; a common value for LoRA fine-tuning
- `fp16=True` — uses 16-bit floating point for speed and memory savings
- `optim="paged_adamw_8bit"` — a memory-efficient optimizer that pages optimizer states to CPU RAM when GPU memory is tight
- `warmup_ratio=0.03` — gradually increases the learning rate for the first 3% of steps to stabilize early training
- `lr_scheduler_type="cosine"` — gradually decreases the learning rate following a cosine curve over training
- `load_best_model_at_end=True` — after training finishes, loads the checkpoint that had the best evaluation loss

---

### Step 12: Save the Model to Google Drive

```python
trainer.save_model("./medical_model_final")
tokenizer.save_pretrained("./medical_model_final")

drive.mount('/content/drive')
shutil.copytree("./medical_model_final",
                "/content/drive/MyDrive/medical_ai_model",
                dirs_exist_ok=True)
```

The trained LoRA adapter weights and tokenizer are saved locally to `./medical_model_final`. Then Google Drive is mounted and the model folder is copied there as a backup. This is important because Colab sessions reset after disconnect, and you would lose everything saved only locally. The model saved here contains only the LoRA adapter weights (a few MB), not the full base model.

---

### Step 13: Load and Test the Fine-tuned Model

```python
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
finetuned = PeftModel.from_pretrained(base, "./medical_model_final")
finetuned.eval()

def ask(question):
    prompt = f"""...<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = finetuned.generate(**inputs, max_new_tokens=300,
                                  temperature=0.3, do_sample=True,
                                  pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()
```

To use the fine-tuned model, we load the original base model again and then layer the LoRA adapter on top using `PeftModel.from_pretrained`. `finetuned.eval()` switches the model to inference mode, disabling dropout. The `ask()` function formats the question into the chat template and calls `finetuned.generate()` to produce a response:

- `max_new_tokens=300` — generates at most 300 tokens in the response
- `temperature=0.3` — low temperature makes responses more focused and deterministic
- `do_sample=True` — enables sampling-based generation (as opposed to greedy decoding)

The generated output includes the full prompt followed by the model's response, so we split on `"assistant"` and take the last part.

---

### Step 14: Add Refusal Training Examples

```python
refusal_examples = [
    {
        "instruction": "What is machine learning?",
        "output": "I am a specialized Medical AI Assistant. I can only answer medical and health-related questions..."
    },
    ...
]

medical_data = medical_data + refusal_examples
```

We add 12 more training examples that cover non-medical questions (technology, history, geography, entertainment, etc.). For each one, the correct answer is a polite refusal explaining that the model only handles medical queries. These examples teach the model to recognize out-of-scope questions and respond appropriately rather than making up an answer.

---

### Step 15: Keyword-Based Hard Filter for Non-Medical Questions

```python
MEDICAL_KEYWORDS = [
    "symptom", "disease", "medicine", "doctor", "patient", "fever",
    "pain", "diagnosis", "treatment", "hospital", ...
]

def is_medical_question(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in MEDICAL_KEYWORDS)

def ask(question):
    if not is_medical_question(question):
        return ("I am a specialized Medical AI Assistant. "
                "I can only answer medical and health-related questions.")
    # ... run model only if medical
```

As a safety layer on top of the model's learned behavior, we add a simple keyword filter. Before even sending the question to the model, the `is_medical_question()` function checks if the question contains any word from the medical keywords list. If no medical keyword is found, the function returns a fixed refusal message immediately without calling the model at all. This makes the refusal fast and deterministic for obvious non-medical questions, regardless of how the model was fine-tuned.

---

## Requirements

- Google Colab with T4 GPU (free tier works)
- Hugging Face account with access to `meta-llama/Llama-3.2-1B-Instruct`
- Hugging Face API token stored as `HF_TOKEN` in Colab secrets

---

## Project Structure

```
medical-ai-finetune/
├── README.md                   # This file
├── finetune_medical_llama3.2_qlora.ipynb      # Main Colab notebook
├── finetune_medical_llama3.2_qlora.py           # Auto-generated training data (24 examples)
```

---

## Key Concepts Summary

| Concept | What It Does |
|---|---|
| QLoRA | Combines 4-bit quantization with LoRA to fine-tune large models on limited GPU memory |
| 4-bit Quantization | Compresses model weights from 16/32-bit floats to 4-bit, reducing VRAM usage by ~4x |
| LoRA | Adds small trainable adapter matrices to frozen model layers instead of updating all weights |
| Gradient Accumulation | Simulates larger batch sizes by accumulating gradients over multiple small batches |
| System Prompt | Instruction given to the model at the start of every conversation to define its role and behavior |
| Tokenization | Converting text into numerical token IDs that the model can process |

---

## Notes

- The model only answers medical questions. Non-medical questions are refused both by the keyword filter and by the model's learned behavior from refusal training examples.
- Always consult a qualified doctor for actual medical diagnosis and treatment. This model is for educational and demonstration purposes only.
- Training takes approximately 15-20 minutes on a T4 GPU with this dataset size.

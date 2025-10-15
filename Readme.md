#  Fine-Tuning Qwen-2.5 Using Lama Factory

This repository demonstrates how to **fine-tune the Qwen-2.5 large language model** using the **Lama Factory** framework.  
It provides a full end-to-end workflow — from data preparation and configuration to training, evaluation, and inference.

---

##  Overview

**Goal:**  
Adapt Qwen-2.5 to your custom dataset or task using efficient fine-tuning methods such as LoRA or adapters, leveraging the modular tools provided by Lama Factory.

**Key Features**
-  Instruction or domain-specific fine-tuning for Qwen-2.5  
-  Configurable training (learning rate, epochs, adapters, LoRA, etc.)  
-  Data preprocessing utilities  
-  Evaluation and inference scripts  
-  Example notebook for experimentation  

---

## 📂 Repository Structure

.
├── Fine_Tuning_Qwen_.ipynb # Main Jupyter notebook for experiments
├── data/ # Datasets (raw / processed)
├── configs/ # Model and training configuration files
│ ├── base_config.yaml
│ └── qwen_adapter_config.yaml
├── scripts/ # CLI scripts for pipeline steps
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
├── src/ # Core logic and helpers
│ ├── model_utils.py
│ ├── trainer.py
│ └── data_loader.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

##  Getting Started

### 1️⃣ Prerequisites
- Python **3.8+**
- CUDA-enabled GPU (≥ 24 GB VRAM recommended)
- Optional: DeepSpeed or distributed training library

---

### 2️⃣ Installation

```bash
git clone https://github.com/youssefsalah224/finetuning-Qwen2.5-using-Lama-Factory.git
cd finetuning-Qwen2.5-using-Lama-Factory
python3 -m venv venv
source venv/bin/activate   # (use venv\Scripts\activate on Windows)
pip install --upgrade pip
pip install -r requirements.txt
3️⃣ Prepare Data
Format your dataset in JSONL or other supported structure (e.g., {"input": "...", "output": "..."}).

Place it inside the data/ folder.

(Optional) Run preprocessing:

bash
Copy code
python scripts/preprocess.py \
  --input_path data/raw.jsonl \
  --output_path data/processed.jsonl \
  --tokenizer_path path/to/qwen-tokenizer
4️⃣ Configure Training
Edit your configuration file (e.g., configs/qwen_adapter_config.yaml) to define:

Model path or checkpoint

Training hyperparameters (lr, batch_size, epochs)

Adapter / LoRA options

Mixed precision mode (fp16 or bf16)

Output checkpoint directory

5️⃣ Run Training
bash
Copy code
python scripts/train.py \
  --config configs/qwen_adapter_config.yaml
Or open and run the notebook:

bash
Copy code
Fine_Tuning_Qwen_.ipynb
6️⃣ Evaluate and Inference
Once training finishes, you can evaluate or generate predictions:

bash
Copy code
python scripts/evaluate.py \
  --checkpoint_path runs/exp1/checkpoint \
  --test_data data/test.jsonl \
  --output_path results/output.jsonl
You can also load your model using Hugging Face’s pipeline or the helper functions in src/model_utils.py.

Tips & Best Practices
Use gradient accumulation to simulate large batch sizes on limited GPUs.

Enable mixed precision (fp16/bf16) for better efficiency.

Regularly save checkpoints and monitor validation metrics.

Consider using early stopping to avoid overfitting.

Track progress via TensorBoard or Weights & Biases.

Expected Results
Fine-tuned models should show improved accuracy, instruction-following ability, or domain relevance compared to the base Qwen-2.5 model.

You can evaluate performance with metrics like:

BLEU / ROUGE (for text generation)

Accuracy / F1 (for classification)

Perplexity (for language modeling)

 Limitations
Requires strong GPU resources for large-scale fine-tuning

Potential catastrophic forgetting on small datasets

Fine-tuned models may inherit biases from training data

 Acknowledgments
Qwen-2.5 by Alibaba Cloud

Lama Factory for the modular fine-tuning toolkit

Hugging Face Transformers and PEFT libraries for model utilities


Citation
If you use this project or build upon it, please cite:

text
Copy code
@misc{youssef2025qwenlama,
  title = {Fine-Tuning Qwen-2.5 Using Lama Factory},
  author = {Salah, Youssef},
  year = {2025},
  url = {https://github.com/youssefsalah224/finetuning-Qwen2.5-using-Lama-Factory}
}


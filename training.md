
## Parameter-Efficient Fine-Tuning (PEFT)

* fine-tune a small number of (extra) model parameters
* keep most parameters of the pretrained LLMs unchanged
* decreasing the computational and storage costs

## LoRA (Low-Rank Adaptation)

* One example of PEFT
* Freezes the pre-trained model weights
* Injects trainable rank decomposition matrices into each layer of the Transformer
* Reducing the number of trainable parameters for downstream tasks


**Example 1**

* Load a pre-trained model to train with PEFT (LoRA)

```bash
pip install peft
```

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM

model_name = "t5-base"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)
...
model.save_pretrained("output_dir") 
```

**Example 2**

* Load the pre-trained PEFT model and continue training

```python
from peft import PeftModel, PeftConfig

peft_model_id = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

```

---

## SFT Trainer

* Supervised fine-tuning (or SFT for short) is a crucial step in RLHF.
* [Transformers](https://huggingface.co/docs/transformers/en/index) provides a Trainer class for fine-tuning pre-trained models.

**TRL Library**

[TRL](https://huggingface.co/docs/trl/en/index) is a full stack library providing a set of tools to train transformer language models with Reinforcement Learning. (e.g. Supervised Fine-tuning step (SFT), Reward Modeling step (RM), Proximal Policy Optimization (PPO)).

```bash
pip install datasets
pip install trl
```

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

```


### More materials to read

* [https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs](https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs)


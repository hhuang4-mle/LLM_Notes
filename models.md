
## Gemma

[Gemma](https://blog.google/technology/developers/gemma-open-models/) is a family of lightweight open models built from the same research and technology used to create the Gemini models by Google DeepMind.

2 Model weights:
* Gemma 2B 
* Gemma 7B 

Training Dataset (key components) of Gemma
* Web Documents: A diverse collection of web text (linguistic styles, topics, and vocabulary, primarily in English).
* Code: syntax and patterns of programming languages
* Mathematics: logical reasoning, symbolic representation, and to address mathematical queries.

**Example**

* Use Gemma with [HuggingFace pipeline](pipeline.md)

```python
from transformers import AutoTokenizer, pipeline
import torch

model = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
model = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = model(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
```


### Fine Tuning Gemma Model

[Source Link](https://huggingface.co/blog/gemma-peft)

* **Part1** - Load the Gemma model and initialise the tokeniser

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
```

* **Part2** - Load the trainer and dataset. Then run the training process.

```python
import transformers
from trl import SFTTrainer
from datasets import load_dataset


data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"Quote: {example['quote'][i]}\nAuthor: {example['author'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)

trainer.train()
```

---

## Llama 2

[Llama 2](https://llama.meta.com/) is a LLM released by Meta. It is a collection of foundation language models ranging from 7B to 70B parameters, with checkpoints fine tuned for chat application.

**Example**

* Use Llama2 with [HuggingFace pipeline](https://huggingface.co/blog/llama2)

```python

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

output = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)


print(output)
```

### Fine-Tuning Llama2

[Source Link](https://www.datacamp.com/tutorial/fine-tuning-llama-2)

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

base_model = "NousResearch/Llama-2-7b-chat-hf"


dataset = load_dataset(guanaco_dataset, split="train")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

```
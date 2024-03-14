
## Hugging Face Pipelines

* The [transformers](https://huggingface.co/docs/transformers/en/index) package by HuggingFace consists of 170 pretrained models across language, vision, audio, and multi-modal modalities. It supports frameworks such as PyTorch, TensorFlow, and JAX.

```bash
pip install transformers
```

The [pipeline](https://huggingface.co/docs/transformers/v4.38.2/en/quicktour#pipeline) provides an abstraction of the complicated code and offer simple API for several tasks such as Text Summarization, Question Answering, Named Entity Recognition, Text Generation etc.

**Example 1**

* Question Answering Pipeline

```python
from transformers import pipeline

qa_pipeline = pipeline(model="deepset/roberta-base-squad2")

qa_pipeline(
    question="Where do I work?",
    context="I work as a Data Scientist at a lab in University of Montreal. I like to develop my own algorithms.",
)

```

**Example 2**

* Named Entity Recognition

```
from transformers import pipeline


ner_classifier = pipeline(
    model="dslim/bert-base-NER-uncased", aggregation_strategy="simple"
)
sentence = "I like to travel in Montreal."
entity = ner_classifier(sentence)
print(entity)
```
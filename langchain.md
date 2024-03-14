
## LangChain

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing LLM applications. 
It provides a simple interface for developers to interact with LLMs. It works like a reductionist wrapper for leveraging LLMs.

* **Example 1** - Use the OpenAI models with LangChain 

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key="YOUR_API_KEY", openai_organization="YOUR_ORGANIZATION_ID")

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
chat.invoke(messages)
```

* **Example 2** - Use the Google AI models with LangChain 

```python
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
result = model.invoke(
    [
        SystemMessage(content="Answer only yes or no."),
        HumanMessage(content="Is apple a fruit?"),
    ]
)
print(result.content)
```

* **Example 3** - Use HuggingFace Pipeline with LangChain

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)
```


### Custom LLM

LangChain allows users to create their own LLM models.

There are only two required things that a custom LLM needs to implement:

* A `_call` method that takes in a string, some optional stop words, and returns a string.
* A `_llm_type` property that returns a string. Used for logging purposes only.

There is a second optional thing it can implement:

* An `_identifying_params` property that is used to help with printing of this class. Should return a dictionary.


* **Example 1** - A simple example of a Custom LLM using LangChain

```python
class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
```

* **Example 2** - An example of implementing a Gemma LLM using LangChain’s Custom LLM wrapper.

[source code link](https://python.langchain.com/docs/modules/model_io/llms/custom_llm)

```python
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()

class GemmaLLM(LLM):
    hf_pipe: Any = None
    pipe_kwargs: Any = None
        
    def __init__(self, hf_pipeline, pipe_kwargs):
        super(GemmaLLM, self).__init__()
        self.hf_pipe = hf_pipeline
        self.pipe_kwargs = pipe_kwargs

    @property
    def _llm_type(self):
        return "Gemma pipeline"

    def _call(self, prompt, **kwargs):
        outputs = self.hf_pipe(
            prompt,
            do_sample=self.pipe_kwargs['do_sample'],
            temperature=self.pipe_kwargs['temperature'],
            top_k=self.pipe_kwargs['top_k'],
            top_p=self.pipe_kwargs['top_p'],
            add_special_tokens=self.pipe_kwargs['add_special_tokens']
        )
        return outputs[0]["generated_text"][len(prompt):]  

    @property
    def _identifying_params(self):
        """Pipeline params"""
        return {"n": self.pipe_kwargs}

hf = GemmaLLM(
	hf_pipeline = pipe,
    pipe_kwargs = {
        'do_sample':True,
        'temperature':0.1,
        'top_k':20,
        'top_p':0.3,
        'add_special_tokens':True
    })
```

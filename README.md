# Databricks Generative AI Inference SDK (wfork)

[![PyPI version](https://img.shields.io/pypi/v/wfork-databricks-genai-inference.svg)](https://pypi.org/project/wfork-databricks-genai-inference/)

The Databricks Generative AI Inference Python library provides a user-friendly python interface to use the Databricks [Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html). Since the library is marked as Apache license, this is the wfork, my fork that provides minor updates while we wait for the official new version that's been a long time coming.

> [!NOTE]
> This SDK was primarily designed for pay-per-token endpoints (`databricks-*`). It has a list of known model names (eg. `dbrx-instruct`) and automatically maps them to the corresponding shared endpoint (`databricks-dbrx-instruct`).
> You can use this with provisioned throughput endpoints, as long as they do not match known model names.
> If there is an overlap, you can use the `DATABRICKS_MODEL_URL_ENV` URL to directly provide an endpoint URL.

This library includes a pre-defined set of API classes `Embedding`, `Completion`, `ChatCompletion` with convenient functions to make API request, and to parse contents from raw json response. 

It also offers a high level `ChatSession` object for easy management of multi-round chat completions, which is especially useful for your next chatbot development.

You can find more usage details in the databricks [SDK onboarding doc](https://docs.databricks.com/en/machine-learning/foundation-models/query-foundation-model-apis.html).

> [!IMPORTANT]  
> They are allegedly preparing to release version 1.0 of the official Databricks GenerativeAI Inference Python library, which will probably be better than this fork. Watch https://pypi.org/project/databricks-genai-inference/ like a hawk for more.

## Installation

```sh
pip install wfork-databricks-genai-inference
```

(note that the step above is different than the original project, but it's the only step different)

## Usage

### Embedding

```python
from databricks_genai_inference import Embedding
```

(note that the import statement has not changed from the original package!)

#### Text embedding

```python
response = Embedding.create(
    model="bge-large-en", 
    input="3D ActionSLAM: wearable person tracking in multi-floor environments")
print(f'embeddings: {response.embeddings[0]}')
```

> [!TIP]  
> You may want to reuse http connection to improve request latency for large-scale workload, code example:

```python
with requests.Session() as client:
    for i, text in enumerate(texts):
        response = Embedding.create(
            client=client,
            model="bge-large-en",
            input=text
        )
```

#### Text embedding (async)

```python
async with httpx.AsyncClient() as client:
    response = await Embedding.acreate(
        client=client,
        model="bge-large-en", 
        input="3D ActionSLAM: wearable person tracking in multi-floor environments")
    print(f'embeddings: {response.embeddings[0]}')
```

#### Text embedding with instruction

```python
response = Embedding.create(
    model="bge-large-en", 
    instruction="Represent this sentence for searching relevant passages:", 
    input="3D ActionSLAM: wearable person tracking in multi-floor environments")
print(f'embeddings: {response.embeddings[0]}')
```

#### Text embedding (batching)

> [!IMPORTANT]  
> Support max batch size of 150

```python
response = Embedding.create(
    model="bge-large-en", 
    input=[
        "3D ActionSLAM: wearable person tracking in multi-floor environments",
        "3D ActionSLAM: wearable person tracking in multi-floor environments"])
print(f'response.embeddings[0]: {response.embeddings[0]}\n')
print(f'response.embeddings[1]: {response.embeddings[1]}')
```

#### Text embedding with instruction (batching)

> [!IMPORTANT]  
> Support one instruction per batch 
> Batch size

```python
response = Embedding.create(
    model="bge-large-en", 
    instruction="Represent this sentence for searching relevant passages:",
    input=[
        "3D ActionSLAM: wearable person tracking in multi-floor environments",
        "3D ActionSLAM: wearable person tracking in multi-floor environments"])
print(f'response.embeddings[0]: {response.embeddings[0]}\n')
print(f'response.embeddings[1]: {response.embeddings[1]}')
```

### Text completion

```python
from databricks_genai_inference import Completion
```

#### Text completion

```python
response = Completion.create(
    model="mpt-7b-instruct",
    prompt="Represent the Science title:")
print(f'response.text:{response.text:}')

```

#### Text completion (async)

```python
async with httpx.AsyncClient() as client:
    response = await Completion.acreate(
        client=client,
        model="mpt-7b-instruct",
        prompt="Represent the Science title:")
    print(f'response.text:{response.text:}')

```

#### Text completion (streaming)

> [!IMPORTANT]  
> Only support batch size = 1 in streaming mode

```python
response = Completion.create(
    model="mpt-7b-instruct", 
    prompt="Count from 1 to 100:",
    stream=True)
print(f'response.text:')
for chunk in response:
    print(f'{chunk.text}', end="")
```

#### Text completion (streaming + async)

```python
async with httpx.AsyncClient() as client:
    response = await Completion.acreate(
        client=client,
        model="mpt-7b-instruct", 
        prompt="Count from 1 to 10:",
        stream=True)
    print(f'response.text:')
    async for chunk in response:
        print(f'{chunk.text}', end="")

```


#### Text completion (batching)

> [!IMPORTANT]  
> Support max batch size of 16

```python
response = Completion.create(
    model="mpt-7b-instruct", 
    prompt=[
        "Represent the Science title:", 
        "Represent the Science title:"])
print(f'response.text[0]:{response.text[0]}')
print(f'response.text[1]:{response.text[1]}')
```

### Chat completion

```python
from databricks_genai_inference import ChatCompletion
```

> [!IMPORTANT]  
> Batching is not supported for `ChatCompletion`

#### Chat completion

```python
response = ChatCompletion.create(model="llama-2-70b-chat", messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Knock knock."}])
print(f'response.text:{response.message:}')
```

#### Chat completion (async)

```python
async with httpx.AsyncClient() as client:
    response = await ChatCompletion.acreate(
        client=client,
        model="llama-2-70b-chat",
        messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Knock knock."}],
    )
    print(f'response.text:{response.message:}')
```

#### Chat completion (streaming)

```python
response = ChatCompletion.create(model="llama-2-70b-chat", messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Count from 1 to 30, add one emoji after each number"}], stream=True)
for chunk in response:
    print(f'{chunk.message}', end="")
```

#### Chat completion (streaming + async)

```python
async with httpx.AsyncClient() as client:
    response = await ChatCompletion.acreate(
        client=client,
        model="llama-2-70b-chat",
        messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Count from 1 to 30, add one emoji after each number"}],
        stream=True,
    )
    async for chunk in response:
        print(f'{chunk.message}', end="")
```

### Chat session

```python
from databricks_genai_inference import ChatSession
```

> [!IMPORTANT]  
> Streaming mode is not supported for `ChatSession`

```python
chat = ChatSession(model="llama-2-70b-chat")
chat.reply("Kock, kock!")
print(f'chat.last: {chat.last}')
chat.reply("Take a guess!")
print(f'chat.last: {chat.last}')

print(f'chat.history: {chat.history}')
print(f'chat.count: {chat.count}')
```

# Aleph Alpha Client

[![PyPI version](https://badge.fury.io/py/aleph-alpha-client.svg)](https://pypi.org/project/aleph-alpha-client/)

Interact with the Aleph Alpha API via Python

> [Documentation of the HTTP API can be found here](https://docs.aleph-alpha.com/api/)

## Installation

The latest stable version is deployed to PyPi so you can install this package via pip.

```sh
pip install aleph-alpha-client
```

## Usage

### Completion Multimodal



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaModel, AlephAlphaClient, CompletionRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

# You need to choose a model with multimodal capabilities for this example.
url = "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png"

image = ImagePrompt.from_url(url)
prompt = Prompt([
    image,
    "Q: What does the picture show? A:",
])
request = CompletionRequest(prompt=prompt, maximum_tokens=20)
result = model.complete(request)

print(result.completions[0].completion)
```


### Evaluation text prompt


```python
from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, EvaluationRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

request = EvaluationRequest(prompt=Prompt.from_text("The api works"), completion_expected=" well")
result = model.evaluate(request)

print(result)

```


### Evaluation Multimodal



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient, AlephAlphaModel, EvaluationRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with multimodal capabilities for this example.
    model_name = "luminous-extended"
)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = Prompt([
    image,
    "Q: What is the name of the store?\nA:",
])
request = EvaluationRequest(prompt=prompt, completion_expected=" Blockbuster Video")
result = model.evaluate(request)

print(result)
```


### Embedding text prompt



```python
from aleph_alpha_client import AlephAlphaModel, AlephAlphaClient, EmbeddingRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

request = EmbeddingRequest(prompt=Prompt.from_text("This is an example."), layers=[-1], pooling=["mean"])
result = model.embed(request)

print(result)
```


### Embedding multimodal prompt



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient, AlephAlphaModel, EmbeddingRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with multimodal capabilities for this example.
    model_name = "luminous-extended"
)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = Prompt([
    image,
    "Q: What is the name of the store?\nA:",
])
request = EmbeddingRequest(prompt=prompt, layers=[-1], pooling=["mean"])
result = model.embed(request)

print(result)
```

### Semantic embedding

#### Symmetric


```python
from typing import Sequence
from aleph_alpha_client import ImagePrompt, AlephAlphaClient, AlephAlphaModel, SemanticEmbeddingRequest, SemanticRepresentation, Prompt
import math
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with multimodal capabilities for this example.
    model_name = "luminous-base"
)

# Texts to compare
texts = [
    "deep learning",
    "artificial intelligence",
    "deep diving",
    "artificial snow",
]

embeddings = []

for text in texts:
    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), representation=SemanticRepresentation.Symmetric)
    result = model.semantic_embed(request)
    embeddings.append(result.embedding)

# Calculate cosine similarities. Can use numpy or scipy or another library to do this
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
# Cosine similarities are in [-1, 1]. Higher means more similar
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_similarity(embeddings[0], embeddings[1])))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_similarity(embeddings[0], embeddings[2])))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[3], cosine_similarity(embeddings[0], embeddings[3])))
```

#### Documents and Query


```python
from typing import Sequence
from aleph_alpha_client import ImagePrompt, AlephAlphaClient, AlephAlphaModel, SemanticEmbeddingRequest, SemanticRepresentation, Prompt
import math
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with multimodal capabilities for this example.
    model_name = "luminous-base"
)

# Documents to search in
documents = [
    # AI wikipedia article
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
    # Deep Learning Wikipedia article
    "Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
    # Deep Diving Wikipedia article
    "Deep diving is underwater diving to a depth beyond the norm accepted by the associated community. In some cases this is a prescribed limit established by an authority, while in others it is associated with a level of certification or training, and it may vary depending on whether the diving is recreational, technical or commercial. Nitrogen narcosis becomes a hazard below 30 metres (98 ft) and hypoxic breathing gas is required below 60 metres (200 ft) to lessen the risk of oxygen toxicity.",
]
# Keyword to search documents with
query = "artificial intelligence"

# Embed Query
request = SemanticEmbeddingRequest(prompt=Prompt.from_text(query), representation=SemanticRepresentation.Query)
result = model.semantic_embed(request)
query_embedding = result.embedding

# Embed documents
document_embeddings = []

for document in documents:
    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(document), representation=SemanticRepresentation.Document)
    result = model.semantic_embed(request)
    document_embeddings.append(result.embedding)

# Calculate cosine similarities. Can use numpy or scipy or another library to do this
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
# Cosine similarities are in [-1, 1]. Higher means more similar
print("Cosine similarity between \"%s\" and \"%s...\" is: %.3f" % (query, documents[0][:10], cosine_similarity(query_embedding, document_embeddings[0])))
print("Cosine similarity between \"%s\" and \"%s...\" is: %.3f" % (query, documents[1][:10], cosine_similarity(query_embedding, document_embeddings[1])))
print("Cosine similarity between \"%s\" and \"%s...\" is: %.3f" % (query, documents[2][:10], cosine_similarity(query_embedding, document_embeddings[2])))
```


### Q&A with a Docx Document



```python
from aleph_alpha_client import Document, AlephAlphaClient, AlephAlphaModel, QaRequest
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with qa support for this example.
    model_name = "luminous-extended"
)

docx_file = "./tests/sample.docx"
document = Document.from_docx_file(docx_file)

request = QaRequest(
    query = "What is a computer program?",
    documents = [document]
)

result = model.qa(request)

print(result)
```


### Q&A with a Text


```python
from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, QaRequest
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with qa support for this example.
    model_name = "luminous-extended"
)

prompt = "In imperative programming, a computer program is a sequence of instructions in a programming language that a computer can execute or interpret."
document = Document.from_text(prompt)

request = QaRequest(
    query = "What is a computer program?",
    documents = [document],
)

result = model.qa(request)

print(result)
```


### Q&A with a multimodal prompt



```python
from aleph_alpha_client import Document, ImagePrompt, AlephAlphaClient, AlephAlphaModel, QaRequest
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    # You need to choose a model with qa support for this example.
    model_name = "luminous-extended"
)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = [image]
document = Document.from_prompt(prompt)

request = QaRequest (
    query = "What is the name of the store?",
    documents = [document]
)

result = model.qa(request)

print(result)
```


### Tokenize a text prompt


```python
from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, TokenizationRequest
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

# You need to choose a model with qa support and multimodal capabilities for this example.
request = TokenizationRequest(prompt="This is an example.", tokens=True, token_ids=True)
response = model.tokenize(request)

print(response)
```


### Detokenize a token IDs into text prompt


```python
from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, DetokenizationRequest
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

# You need to choose a model with qa support and multimodal capabilities for this example.
request = DetokenizationRequest(token_ids=[1730, 387, 300, 4377, 17])
response = model.detokenize(request)

print(response)
```


## Testing

Tests use pytests with (optional) coverage plugin. Install the locally cloned repo in editable mode with:

```bash
pip install -e .[test]
```

**Tests make api calls that reduce your quota!**

### Run tests

Tests can be run using pytest. Make sure to create a `.env` file with the following content:

```env
# test settings
TEST_API_URL=https://test.api.aleph-alpha.com
TEST_MODEL=luminous-base
TEST_TOKEN=your_token
```

Instead of a token username and password can be used.

```env
# test settings
TEST_API_URL=https://api.aleph-alpha.com
TEST_MODEL=luminous-base
TEST_USERNAME=your_username
TEST_PASSWORD=your_password
```

* A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)
* A subset of tests can be selected by pointing to the module within tests

```bash
# run all tests, output coverage report of aleph_alpha_client module in terminal
pytest --cov-report term --cov=aleph_alpha_client tests
pytest tests -v # start verbose
```

If an html coverage report has been created a simple http server can be run to serve static files.

```bash
python -m http.server --directory htmlcov 8000
```

## Update README

> Do not change the README.md directly as it is generated from readme.ipynb 

To update the readme, do the following:

1. `pip install -e .[dev]`

2. Edit the notebook in your favorite jupyter editor and run all python cells to verify that the code examples still work.

3. To generate a new README.md first remove all output cells from the Jupyter notebook and then execute the command: `jupyter nbconvert --to markdown readme.ipynb --output README.md`


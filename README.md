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
from aleph_alpha_client import ImagePrompt, AlephAlphaClient
from aleph_alpha_client.completion import CompletionRequest
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with multimodal capabilities for this example.
model = "luminous-base"
url = "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png"

image = ImagePrompt.from_url(url)
prompt = [
    image,
    "Q: What does the picture show? A:",
]
request = CompletionRequest(prompt=prompt, maximum_tokens=20)
result = client.complete(model, request=request)

print(result.completions[0]["completion"])
```


### Evaluation text prompt


```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

model = "luminous-base"
prompt = "The api works"
result = client.evaluate(model, prompt=prompt, completion_expected=" well")

print(result)

```


### Evaluation Multimodal



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with multimodal capabilities for this example.
model = "luminous-base"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = [
    image,
    "Q: What is the name of the store?\nA:",
]

result = client.evaluate(model, prompt=prompt, completion_expected=" Blockbuster Video")

print(result)
```


### Embedding text prompt



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient, EmbeddingRequest
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

model = "luminous-base"
request = EmbeddingRequest(prompt="This is an example.", layers=[-1], pooling=["mean"])
result = client.embed(model, request)

print(result)
```


### Embedding multimodal prompt



```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with multimodal capabilities for this example.
model = "luminous-base"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = [
    image,
    "Q: What is the name of the store?\nA:",
]
request = EmbeddingRequest(prompt=prompt, layers=[-1], pooling=["mean"])
result = client.embed(model, request)

print(result)
```


### Q&A with a Docx Document



```python
from aleph_alpha_client import Document, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with qa support for this example.
model = "luminous-extended"

query = "What is a computer program?"
docx_file = "./tests/sample.docx"
document = Document.from_docx_file(docx_file)
documents = [document]

result = client.qa(model, query=query, documents=documents, maximum_tokens=64)

print(result)
```


### Q&A with a Text


```python
from aleph_alpha_client import Document, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with qa support for this example.
model = "luminous-extended"

query = "What is a computer program?"
prompt = "In imperative programming, a computer program is a sequence of instructions in a programming language that a computer can execute or interpret."
document = Document.from_text(prompt)
documents = [document]

result = client.qa(model, query=query, documents=documents, maximum_tokens=64)

print(result)
```


### Q&A with a multimodal prompt



```python
from aleph_alpha_client import Document, ImagePrompt, AlephAlphaClient
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with qa support and multimodal capabilities for this example.
model = "luminous-extended"

query = "What is the name of the store?"
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = [image]
document = Document.from_prompt(prompt)
documents = [document]

result = client.qa(model, query=query, documents=documents, maximum_tokens=64)

print(result)
```


### Tokenize a text prompt


```python
from aleph_alpha_client import Document, ImagePrompt, AlephAlphaClient
from aleph_alpha_client.tokenization import TokenizationRequest
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with qa support and multimodal capabilities for this example.
model = "luminous-extended"
request = TokenizationRequest(prompt="This is an example.", tokens=True, token_ids=True)
response = client.tokenize(model, request=request)

print(response)
```


### Detokenize a token IDs into text prompt


```python
from aleph_alpha_client import Document, ImagePrompt, AlephAlphaClient
from aleph_alpha_client.detokenization import DetokenizationRequest
import os

client = AlephAlphaClient(
    host="https://api.aleph-alpha.com",
    token=os.getenv("AA_TOKEN")
)

# You need to choose a model with qa support and multimodal capabilities for this example.
model = "luminous-extended"
request = DetokenizationRequest(token_ids=[1730, 387, 300, 4377, 17])
response = client.detokenize(model, request=request)

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

1. `pip install -r requirements-dev.txt`

2. To update the readme edit the notebook in your favorite jupyter editor and run all python cells to verify that the code examples still work.

3. To generate a new README.md first remove all output cells from the Jupyter notebook and then execute the command: `jupyter nbconvert --to markdown readme.ipynb --output README.md`


# Aleph Alpha Client

<p align="center">
    <img src="https://i.imgur.com/FSM2NNV.png" width="50%" />
</p>

[![License](https://img.shields.io/crates/l/aleph-alpha-client)](https://github.com/Aleph-Alpha/aleph-alpha-client/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/aleph-alpha-client.svg)](https://pypi.org/project/aleph-alpha-client/)
[![Documentation Status](https://readthedocs.org/projects/aleph-alpha-client/badge/?version=latest)](https://aleph-alpha-client.readthedocs.io/en/latest/?badge=latest)

Python client for the [Aleph Alpha](https://aleph-alpha.com) API.

## Usage

### Synchronous Client

```python
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt

client = Client(token=os.getenv("AA_TOKEN"))
request = CompletionRequest(
    prompt=Prompt.from_text("Provide a short description of AI:"),
    maximum_tokens=64,
)
response = client.complete(request, model="luminous-extended")

print(response.completions[0].completion)
```

### Asynchronous Client

```python
import os
from aleph_alpha_client import AsyncClient, CompletionRequest, Prompt

# Can enter context manager within an async function
async with AsyncClient(token=os.environ["AA_TOKEN"]) as client:
    request = CompletionRequest(
        prompt=Prompt.from_text("Provide a short description of AI:"),
        maximum_tokens=64,
    )
    response = await client.complete(request, model="luminous-base") 
    print(response.completions[0].completion)
```

### Interactive Examples

This table contains interactive code examples, further exercises can be found in the [examples repository](https://github.com/Aleph-Alpha/examples).
| Template | Description | Internal Link | Colab Link |
|----------|-------------|---------------| -----------|
| 1 | Calling the API | [Template 1](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/01_using_client.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/01_using_client.ipynb)|
| 2 | Simple completion | [Template 2](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/02_prompting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/02_prompting.ipynb)|
| 3 | Simple search | [Template 3](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/03_simple_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/03_simple_search.ipynb)|
| 4 | Symmetric and Asymmetric Search | [Template 4](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/04_semantic_search.ipynb)|
| 5 | Hidden Embeddings | [Template 5](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/05_hidden_embeddings.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/05_hidden_embeddings.ipynb)|
| 6 | Combining functionalities | [Template 6](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/06_combining_functionalities.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/06_combining_functionalities.ipynb)|

## Installation

The latest stable version is deployed to PyPi so you can install this package via pip.

```sh
pip install aleph-alpha-client
```

Get started using the client by first [creating an account](https://app.aleph-alpha.com/signup). Afterwards head over to [your profile](https://app.aleph-alpha.com/profile) to create an API token. Read more about how you can manage your API tokens [here](https://docs.aleph-alpha.com/docs/account).

## Links

- [HTTP API Docs](https://docs.aleph-alpha.com/api/)
- [Interactive Playground](https://app.aleph-alpha.com/playground/)

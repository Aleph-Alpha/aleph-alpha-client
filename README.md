# Aleph Alpha Client

<p align="center">
    <img src="https://i.imgur.com/FSM2NNV.png" width="50%" />
</p>

[![Licence](https://img.shields.io/crates/l/aleph-alpha-client)](https://github.com/Aleph-Alpha/aleph-alpha-client/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/aleph-alpha-client.svg)](https://pypi.org/project/aleph-alpha-client/)

Python client for the [Aleph Alpha](https://aleph-alpha.com) API.

## Usage

### Text Completion

```python
from aleph_alpha_client import AlephAlphaModel, AlephAlphaClient, CompletionRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    model_name = "luminous-extended"
)

prompt = Prompt([
    "Provide a short description of AI:",
])
request = CompletionRequest(prompt=prompt, maximum_tokens=20)
result = model.complete(request)

print(result.completions[0].completion)
```

### Interactive Examples

This table contains interactive code examples, further exercises can be found in the [examples repository](https://github.com/Aleph-Alpha/examples).
| Template | Description | Internal Link | Colab Link |
|----------|-------------|---------------| -----------|
| 1 | Calling the API | [Template 1](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/01_calling_api.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/01_calling_api.ipynb)|
| 2 | Simple completion | [Template 2](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/02_simple_completion.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/02_simple_completion.ipynb)|
| 3 | Simple search | [Template 3](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/03_simple_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/03_simple_search.ipynb)|
| 4 | Symmetric and Asymmetric Search | [Template 4](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/04_symmetric_and_asymmetric_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/04_symmetric_and_asymmetric_search.ipynb)|
| 5 | Hidden Embeddings | [Template 5](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/05_hidden_embeddings.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/05_hidden_embeddings.ipynb)|
| 6 | Combining functionalities | [Template 6](https://github.com/Aleph-Alpha/examples/blob/main/boilerplate/06_combining_functionalities.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aleph-Alpha/examples/blob/main/boilerplate/06_combining_functionalities.ipynb)|

## Installation

The latest stable version is deployed to PyPi so you can install this package via pip.

```sh
pip install aleph-alpha-client
```

Get started using the client by first [creating an account](https://app.aleph-alpha.com/signup). Afterwards head over to [your profile](https://app.aleph-alpha.com/profile) to create an API token. Read more about how you can manage your API tokens [here](https://docs.aleph-alpha.com/changelog/2022/10/26/token-management/).

## Links

- [HTTP API Docs](https://docs.aleph-alpha.com/api/)
- [Interactive Playground](https://app.aleph-alpha.com/playground/)

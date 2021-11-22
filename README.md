# Aleph Alpha Client

[![PyPI version](https://badge.fury.io/py/aleph-alpha-client.svg)](https://pypi.org/project/aleph-alpha-client/)

Interact with the Aleph Alpha API via Python

> [Documentation of the HTTP API can be found here](https://github.com/Aleph-Alpha/aleph-alpha-client/blob/master/API_Docs.md)

## Installation

The latest stable version is deployed to PyPi so you can install this package via pip.

```sh
pip install aleph-alpha-client
```

## Usage

### Completion Multimodal

```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient

client = AlephAlphaClient(
    host="https://api.aleph-alpha.de",
    token="<your token>
)

# You need to choose a model with multimodal capabilities for this example.
model = "EUTranMultimodal"
url = "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png"

image = ImagePrompt.from_url(url)
prompt = [
    image,
    "Q: What does the picture show? A:",
]
result = client.complete(model, prompt=prompt, maximum_tokens=20)

print(result["completions"][0]["completion"])
```

### Evaluation text prompt

```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient

client = AlephAlphaClient(
    host="https://api.aleph-alpha.de",
    token="<your token>
)

model = "EleutherAI/gpt-neo-2.7B"
prompt = "The api works"
result = client.evaluate(model, prompt=prompt, completion_expected=" well")

print(result)
```

### Evaluation Multimodal

```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient

client = AlephAlphaClient(
    host="https://api.aleph-alpha.de",
    token="<your token>
)

# You need to choose a model with multimodal capabilities for this example.
model = "EUTranMultimodal"

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
from aleph_alpha_client import ImagePrompt, AlephAlphaClient

client = AlephAlphaClient(
    host="https://api.aleph-alpha.de",
    token="<your token>
)

model = "EleutherAI/gpt-neo-2.7B"
prompt = "This is an example."
result = client.embed(model, prompt=prompt, layers=[-1], pooling=["mean"])

print(result)
```

### Embedding multimodal prompt

```python
from aleph_alpha_client import ImagePrompt, AlephAlphaClient

client = AlephAlphaClient(
    host="https://api.aleph-alpha.de",
    token="<your token>
)

# You need to choose a model with multimodal capabilities for this example.
model = "EUTranMultimodal"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/2008-09-24_Blockbuster_in_Durham.jpg/330px-2008-09-24_Blockbuster_in_Durham.jpg"
image = ImagePrompt.from_url(url)
prompt = [
    image,
    "Q: What is the name of the store?\nA:",
]

result = client.embed(model, prompt=prompt, layers=[-1], pooling=["mean"])

print(result)
```

## Endpoints

### Complete

generate completions from a prompt

#### Parameters

**model** (str, required)

Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.  
see `available_models()`

**prompt** (str or multimodal list*, optional, default "")

The text to be completed. Unconditional completion can be started with an empty string (default). The prompt may contain a zero shot or few shot task.

**hosting** (str, optional, default "cloud"):

Specifies where the computation will take place. This defaults to "cloud", meaning that it can be
executed on any of our servers. An error will be returned if the specified hosting is not available.
Check available_models() for available hostings.

**maximum_tokens** (int, optional, default 64)

The maximum number of tokens to be generated. Completion will terminate after the maximum number of tokens is reached. Increase this value to generate longer texts. A text is split into tokens. Usually there are more tokens than words. The summed number of tokens of prompt and maximum_tokens depends on the model (for EleutherAI/gpt-neo-2.7B, it may not exceed 2048 tokens).

**temperature** (float, optional, default 0.0)

A higher sampling temperature encourages the model to produce less probable outputs ("be more creative"). Values are expected in a range from 0.0 to 1.0. Try high values (e.g. 0.9) for a more "creative" response and the default 0.0 for a well defined and repeatable answer.

It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

**top_k** (int, optional, default 0)

Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely options. A value larger than 1 encourages the model to be more creative. Set to 0 if repeatable output is to be produced.

It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

**top_p** (float, optional, default 0.0)

Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p. Set to 0.0 if repeatable output is to be produced.

It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

**presence_penalty** (float, optional, default 0.0)

The presence penalty reduces the likelihood of generating tokens that are already present in the text. Presence penalty is independent of the number of occurences. Increase the value to produce text that is not repeating the input.

An operation of like the following is applied:
    logits[t] -> logits[t] - 1 * penalty

where logits[t] is the logits for any given token. Note that the formula is independent of the number of times that a token appears in context_tokens.

**frequency_penalty** (float, optional, default 0.0)

The frequency penalty reduces the likelihood of generating tokens that are already present in the text. Presence penalty is dependent on the number of occurences of a token.

An operation of like the following is applied:
    logits[t] -> logits[t] - count[t] * penalty

where logits[t] is the logits for any given token and count[t] is the number of times that token appears in context_tokens

**repetition_penalties_include_prompt** (bool, optional, default False)

Flag deciding whether presence penalty or frequency penalty are applied to the prompt and completion (True) or only the completion (False)

**use_multiplicative_presence_penalty** (bool, optional, default True)

Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False). This changes the formula stated for presence and frequency penalty.

**best_of** (int, optional, default None)

best_of number of completions are created on server side. The completion with the highest log probability per token is returned. If the parameter n is larger than 1 more than 1 (n) completions will be returned. best_of must be strictly greater than n.

**n** (int, optional, default 1)

Number of completions to be returned. If only the argmax sampling is used (temperature, top_k, top_p are all default) the same completions will be produced. This parameter should only be increased if a random sampling is chosen.

**logit_bias** (dict mapping token ids to score, optional, default None)

The logit bias allows to influence the likelihood of generating tokens. A dictionary mapping token ids (int) to a bias (float) can be provided. Such bias is added to the logits as generated by the model.

**log_probs** (int, optional, default None)

Number of top log probabilities to be returned for each generated token. Log probabilities may be used in downstream tasks or to assess the model's certainty when producing tokens.

No log probs are returned if set to None.
Log probs of generated tokens are returned if set to 0.
Log probs of generated tokens and top n logprobs are returned if set to n.

**stop_sequences** (List(str), optional, default None)

List of strings which will stop generation if they're generated. Stop sequences may be helpful in structured texts.

Example: In a question answering scenario a text may consist of lines starting with either "Question: " or "Answer: " (alternating). After producing an answer, the model will be likely to generate "Question: ". "Question: " may therfore be used as stop sequence in order not to have the model generate more questions but rather restrict text generation to the answers.

**tokens** (bool, optional, default False)

Flag indicating whether individual tokens of the completion are to be returned (True) or whether solely the generated text (i.e. the completion) is sufficient (False).

#### Return value

The return value of a completion contains the following fields:

**id**: unique identifier of a task for traceability

**model_version**: model name and version (if any) of the used model for inference

**completions**: list of completions; may contain only one entry if no more are requested (see parameter n)

**log_probs**: list with a dictionary for each generated token. The dictionary maps the keys' tokens to the respective log probabilities. This field is only returned if requested with the parameter "log_probs".

**completion**: generated completion on the basis of the prompt

**completion_tokens**: completion split into tokens. This field is only returned if requested with the parameter "tokens".

**finish_reason**: reason for termination of generation. This may be a stop sequence or maximum number of tokens reached.

**message**: an optional message by the system. This may contain warnings or hints.

(unconditional completion, prompt is empty)

```json
{
    "id": "6b17dd34-5dc0-4794-aacf-263311965178",
    "model_version": "EleutherAI/gpt-neo-2.7B",
    "completions": [
        {
            "log_probs": null,
            "completion": "Antidote for acute serine in superoxide anion from depycoid and lipopolysaccharide (PE) purified preparations by gel filtration.\nDepycoid and lipopolysaccharide (LPS) isolated from Klepida capers were used as selective antagonists for 8-",
            "completion_tokens": null,
            "finish_reason": null,
            "message": null
        },
        "..."
    ]
}
```

### Evaluate

Evaluates the model's likelihood to produce a completion given a prompt.

#### Parameters

**model** (str, required)

Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.
see `available_models()`

**completion_expected** (str, required)

The ground truth completion expected to be produced given the prompt.

**prompt** (str or multimodal list*, optional, default "")

The text to be completed. Unconditional completion can be used with an empty string (default). The prompt may contain a zero shot or few shot task.

**hosting** (str, optional, default "cloud"):

Specifies where the computation will take place. This defaults to "cloud", meaning that it can be
executed on any of our servers. An error will be returned if the specified hosting is not available.
Check available_models() for available hostings.

#### Return value

The return value of an evaluation contains the following fields:

**id**: unique identifier of a task for traceability

**model_version**: model name and version (if any) of the used model for inference

**message**: an optional message by the system. This may contain warnings or hints.

**result**: dictionary with result metrics of the evaluation

**log_probability**: log probability of producing the expected completion given the prompt. This metric refers to all tokens and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.

**log_perplexity**: log perplexity associated with the expected completion given the prompt. This metric refers to all tokens and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.

**log_perplexity_per_token**: log perplexity associated with the expected completion given the prompt normalized for the number of tokens. This metric computes an average per token and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.

**log_perplexity_per_character**: log perplexity associated with the expected completion given the prompt normalized for the number of characters. This metric is independent of any tokenizer. It can be directly compared among models with different tokenizers.

**correct_greedy**: Flag indicating whether a greedy completion would have produced the expected completion.

**token_count**: Number of tokens in the expected completion.

**character_count**: Number of characters in the expected completion.

**completion**: argmax completion given the input consisting of prompt and expected completion. This may be used as an indicator of what the model would have produced. As only one single forward is performed an incoherent text could be produced especially for long expected completions.  

prompt: "The api works"
completion_expected: " well."

```json
{
    "id": "e0db3dfb-82b0-4554-bb35-e1e124b1c0ee",
    "model_version": "EleutherAI/gpt-neo-2.7B",
    "message": null,
    "result": {
        "log_probability": -5.3242188,
        "log_perplexity": -5.3242188,
        "log_perplexity_per_token": -2.6621094,
        "log_perplexity_per_character": -0.88720703,
        "correct_greedy": false,
        "token_count": 2,
        "character_count": 6,
        "completion": " fine,"
    }
}
```

### Embed

Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

#### Parameters

**model** (str, required)

Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.
see `available_models()`

**prompt** (str or multimodal list*, required)

The text to be embedded.

**layers** (List[int], required)

A list of layer indices from which to return embeddings.

* Index 0 corresponds to the word embeddings used as input to the first transformer layer
* Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
* Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last layer etc.

**hosting** (str, optional, default "cloud"):

Specifies where the computation will take place. This defaults to "cloud", meaning that it can be
executed on any of our servers. An error will be returned if the specified hosting is not available.
Check available_models() for available hostings.

**tokens** (bool, optional, default False)

Flag indicating whether the tokenized prompt is to be returned (True) or not (False)

**pooling** (List[str])

Pooling operation to use.

Pooling operations include:

* mean: aggregate token embeddings across the sequence dimension using an average
* max: aggregate token embeddings across the sequence dimension using a maximum
* last_token: just use the last token
* abs_max: aggregate token embeddings across the sequence dimension using a maximum of absolute values

#### Return value

The return value of an embed contains the following fields:

**id**: unique identifier of a task for traceability

**model_version**: model name and version (if any) of the used model for inference

**message**: an optional message by the system. This may contain warnings or hints.

**embeddings**:
a dict with layer names as keys and and pooling output as values. A pooling output is a dict with pooling operation as key and a pooled embedding (list of floats) as values
  
**tokens**: a list of tokens

example for pooling

```json
{
    "id": "e0db3dfb-82b0-4554-bb35-e1e124b1c0ee",
    "model_version": "EleutherAI/gpt-neo-2.7B",
    "message": null,
    "embeddings": {
       "layer_0": {
           "max": [1,0, 2.0, "..."],
           "mean": [1,0, 2.0, "..."],
       }
    },
    "tokens": ["a", "...",  "z"]
}
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
TEST_API_URL=https://api.aleph-alpha.de
TEST_MODEL=EleutherAI/gpt-neo-2.7B
TEST_TOKEN=your_token
```

Instead of a token username and password can be used.

```env
# test settings
TEST_API_URL=https://api.aleph-alpha.de
TEST_MODEL=EleutherAI/gpt-neo-2.7B
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

# Changelog

## 3.2.0

- Add batch_semantic_embed method for processing batches of semantic embeddings

## 3.1.5

- Remove internal search endpoint again (was only accessible for internal users).

## 3.1.4

### Bug fixes

- Do not send max_answers for QA by default, use API's default instead.

## 3.1.3

### Bug fixes

- Add missing import of **PromptGranularity** in *__init__.py*.

## 3.1.2

### Bug fixes

- **AsyncClient**: Retry on ClientConnectionErrors as well.

## 3.1.1

### Bug Fixes

`PromptGranularity` for `ExplanationRequest`s is now an enum. This was previously just a type alias
for a union of literals but we felt that it would be more natural to have a dedicated enum.

## 3.1.0

### Features

### New `.explain()` method 🎉

Better understand the source of a completion, specifically on how much each section of a prompt impacts the completion.

To get started, you can simply pass in a prompt you used with a model and the completion the model gave and generate an explanation:

```python
from aleph_alpha_client import Client, CompletionRequest, ExplanationRequest, Prompt

client = Client(token=os.environ["AA_TOKEN"])
prompt = Prompt.from_text("An apple a day, ")
model_name = "luminous-extended"

# create a completion request
request = CompletionRequest(prompt=prompt, maximum_tokens=32)
response = client.complete(request, model=model_name)

# generate an explanation
request = ExplanationRequest(prompt=prompt, target=response.completions[0].completion)
response = client.explain(request, model=model_name)
```

To visually see the results, you can also use this in our [Playground](https://app.aleph-alpha.com/playground/explanation).

We also have more [documentation and examples](https://docs.aleph-alpha.com/docs/tasks/explain/) available for you to read.

### AtMan (Attention Manipulation)

Under the hood, we are leveraging the method from our [AtMan paper](https://arxiv.org/abs/2301.08110) to help generate these explanations. And we've also exposed these controls anywhere you can submit us a prompt!

So if you have other use cases for attention manipulation, you can pass these AtMan controls as part of your prompt items.

```python
from aleph_alpha_client import Prompt, Text, TextControl

Prompt([
  Text("Hello, World!", controls=[TextControl(start=0, length=5, factor=0.5)]),
  Image.from_url(
    "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png",
    controls=[ImageControl(top=0.25, left=0.25, height=0.5, width=0.5, factor=2.0)]
  )
])
```

For more information, check out our [documentation and examples](https://docs.aleph-alpha.com/docs/explainability/attention-manipulation/).

## 3.0.0

### Breaking Changes

- Removed deprecated `AlephAlphaClient` and `AlephAlphaModel`. Use `Client` or `AsyncClient` instead.
- Removed deprecated `ImagePrompt`. Import `Image` instead for image prompt items.
- New Q&A interface. We've improved the Q&A implementation, and most parameters are no longer needed.
  - You only need to specify your documents, a query, and (optional) the max number of answers you want to receive.
  - You no longer specify a model.
- Removed "model" parameter from summarize method
- Removed "model_version" from `SummarizationResponse`

## 2.17.0

### Features

- Allow specifying token overlap behavior in AtMan by @benbrandt in https://github.com/Aleph-Alpha/aleph-alpha-client/pull/106

### Bug Fixes

- Better handle case when Prompt is supplied a string instead of a list by @benbrandt in https://github.com/Aleph-Alpha/aleph-alpha-client/pull/107

### Experimental

- New Explain interface for internal testing by @ahartel and @benbrandt in https://github.com/Aleph-Alpha/aleph-alpha-client/pull/97 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/98 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/99 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/100 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/101 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/102 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/103 https://github.com/Aleph-Alpha/aleph-alpha-client/pull/104

## 2.16.1

- AsyncClient now respects http proxy env variables, as Client did all the time
- Update examples links in Readme.md

## 2.16.0

- Add Image.from_image_source

## 2.15.0

- Add completion parameter: repetition_penalties_include_completion, raw_completion
- Fix json deserialization bug: Ignore additional unknown fields in json

## 2.14.0

- Add attention manipulation parameters for images in multimodal prompts

## 2.13.0

- Add attention manipulation parameters on character level

## 2.12.0

- Introduce offline tokenizer
- Add method `models` to Client and AsyncClient to list available models
- Fix docstrings for `complete` methods with respect to Prompt construction
- Minor docstring fix for `evaulate` methods

## 2.11.1

- fix complete in deprecated client: pass None-lists as empty list

## 2.11.0

- add completion parameters: completion_bias_inclusion, completion_bias_inclusion_first_token_only
  completion_bias_exclusion, completion_bias_exclusion_first_token_only

## 2.10.0

- add completion parameters: minimum_tokens, echo, use_multiplicative_frequency_penalty,
  sequence_penalty, sequence_penalty_min_length, use_multiplicative_sequence_penalty

## 2.9.2

- fix type hint in DetokenizationResponse

## 2.9.1

Rerelease to fix tag issue

## 2.9.0

- Adjust default request timeouts
- Fix QA Beta bug in AsyncClient
- Remove experimental checkpoint features

## 2.8.1

- Fix documentation for Client and AsyncClient for readthedocs.io

## 2.8.0

- Add nice flag to clients to indicate lower priority

## 2.7.1

- Add beta flag for QA requests

## 2.7.0

- Add manual on readthedocs
- Increase number of retries and expose parameters

## 2.6.1

- Introduce search endpoint for internal users

## 2.6.0

- Introduce Client
- Add DeprecationWarning for AlephAlphaClient and AlephAlphaModel (to be removed in 3.0.0)

## 2.5.0

- Introduce AsyncClient
- Reworked Readme.md to link to extended examples and Google Colab Notebooks

## 2.4.4

- Fix: ImagePrompt.from_url raises if status-code not OK

## 2.4.3

- Fix: Dependency `urllib` is specified to be at least of version `1.26`.
- Add constructor `AlephAlphaModel::from_model_name`.

## 2.4.2

- Fix: Dependency `requests` is specified to be at least of version `2.28`.

## 2.4.1

### Python support

- Minimal supported Python version is now 3.7
- Previously we only supported version 3.8

## 2.4.0

### New feature

- Internal clients can now select a checkpoints directly and don't have to select a model that processes their requests
- Replaced fields `normalize`, `square_outputs` and `prompt_explain_indices` with `directional` in hidden explain endpoint

## 2.3.0

### New feature

- Summarization of Documents

## 2.2.4

### Documentation

- Update documentation for `hosting` parameter

## 2.2.3

### Bugfix

- Remove `message` field from CompletionResult

## 2.2.2

### Bugfix

- Document `hosting` parameter.
- The hosting parameter determines in which datacenters the request may be processed.
- Currently, we only support setting it to "aleph-alpha", which allows us to only process the request in our own datacenters.
- Not setting this value, or setting it to null, allows us to process the request in both our own as well as external datacenters.

## 2.2.1

### Bugfix

- Restore original error handling of HTTP status codes to before 2.2.0
- Add dedicated exception BusyError for status code 503

## 2.2.0

### New feature

- Retry failed HTTP requests via urllib for status codes 408, 429, 500, 502, 503, 504

## 2.1.0

### New feature

- Add new parameters to control how repetition penalties are applied for completion requests (see [docs](https://docs.aleph-alpha.com/api/#/paths/~1complete/post) for more information):
  - `penalty_bias`
  - `penalty_exceptions`
  - `penalty_exceptions_include_stop_sequences`

## 2.0.0

### Breaking change

- Make hosting parameter optional in semantic_embed on client. Changed order of parameters `hosting` and `request`.
  Should not be an issue if you're not using semantic_embed from the client directly or if you're using keyword args.

### Experimental feature

- Add experimental penalty parameters for completion

## 1.7.1

- Improved handling of text-based Documents in Q&A

## 1.7.0

- Introduce `semantic_embed` endpoint on client and model.
- Introduce timeout on client

## 1.6.0

- Introduce AlephAlphaModel as a more convenient alternative to direct usage of AlephAlphaClient

## 1.1.0

- Support for sending images to multimodal Models.

## 1.0.0

- Initial Release

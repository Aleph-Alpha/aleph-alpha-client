# Changelog

## 11.5.1

- Add `limit` parameter for `AsyncClient`, defaults to 100. This controls the maximum number of concurrent connections to the API.

## 11.5.0

- Add reranking support

## 11.4.0
- Tool calling for chat tasks where the model supports it

## 11.3.0

- Drop support for python3.9

## 11.2.0

- Add support for pydantic objects for structured output response format
- Add additional fields for OpenAI compatibility wrt structured output

## 11.1.0

- Fix API for response_format

## 11.0.0

- enable `encoding_format` parameter for `/embeddings` endpoints
- allow `input` for `/embeddings` to be of other types as well
- fix typo: `EmbeddingV2ReponseData` -> `EmbeddingV2ResponseData`

## 10.6.2

- Fix typing and attribute exports

## 10.6.1

- Fix imports for OpenAI compatible embeddings

## 10.6.0

- Add support for OpenAI compatible `/embeddings` endpoint.

## 10.5.1

- Enable using TextMessage objects in multi-turn chats

## 10.5.0

- Remove default values for top_p, top_k and temperature as not supported by all models

## 10.4.0

- Add support for multimodal prompts (images only)

## 10.3.0

- Add support for translation

## 10.2.2

Add support for Python 3.13

## 10.2.1

- Fix structured output path

## 10.2.0

- Add support for JSON structured output for chat requests

## 10.1.0

- Add support for user-defined steering concepts

## 10.0.1

- Use `uv` for packaging

## 10.0.0

- `finish_reason` attribute of `ChatResponse` is now an instance of `FinishReason` enum
- `complete_with_streaming` yields `FinishReason` as a stream item
- Extra groups `dev`, `types`, `test` become dev-dependencies

## 9.1.0

- Allow steering capabilities for client-submitted chat completion requests

## 9.0.0

- Remove deprecated QA and Summarization functionality
- Remove references to luminous-extended model

## 8.1.0

### Python support

- Minimal supported Python version is now 3.9

## 8.0.0

- Remove default value for `host` parameter in `Client` and `AsyncClient`. Passing a value for
  the `host` is now required.

## 7.6.0

- Add `instructable_embed` to `Client` and `AsyncClient`

## 7.5.1

- Add fallback mechanism for figuring out the version locally.

## 7.5.0

- Add support for chat endpoint to `Client` and `AsyncClient`

## 7.4.0

- Add `complete_with_streaming` to `AsyncClient` to support completion endpoint with streaming

## 7.3.0

- Maximum token attribute of `CompletionRequest` defaults to None

## 7.2.0

### Python support

- Minimal supported Python version is now 3.8
- Dependency `aiohttp` is specified to be at least of version `3.10`.

## 7.1.0

- Introduce support for internal feature 'tags'
- Export BusyError

## 7.0.1

- Fixed a bug in pydoc for Prompt and Token

## 7.0.0

- Added  `num_tokens_prompt_total` to `EvaluationResponse`
- HTTP API version 1.16.0 or higher is required.

## 6.0.0

- Added `num_tokens_prompt_total` to the types below.
  This is a breaking change since `num_tokens_prompt_total` is mandatory.
  - `EmbeddingResponse`
  - `SemanticEmbeddingResponse`
  - `BatchSemanticEmbeddingResponse`
- HTTP API version 1.15.0 or higher is required.

## 5.0.0

- Added `num_tokens_prompt_total` and `num_tokens_generated` to `CompletionResponse`. This is a
  breaking change as these were introduced as mandatory parameters rather than optional ones.
- HTTP API version 1.14.0 or higher is required.

## 4.1.0

- Added `verify_ssl` flag, so you can disable SSL checking for your sessions.

## 4.0.0

- Turned all NamedTuples into Dataclasses. Even if this is technically a breaking change
  probably not much source code is actually affected as constructing instances still behaves
  the same.

## 3.5.1

- Fix failing serialization of Prompt-based Documents in QA requests.
  Documents should also be constructible from actual Prompts and not only from sequences

## 3.5.0

- Deprecation of `qa` and `summarization` methods on `Client` and `AsyncClient`. New methods of processing these tasks will be released before they are removed in the next major version.

## 3.4.2

- Full release for exporting type hints

## 3.4.2a1

- Alpha release for exporting type hints

## 3.4.1

- `PromptTemplate` now resets cached non-text items after generating prompt

## 3.4.0

- `PromptTemplate` now supports embedding full prompts and tokens

## 3.3.1

- Mismatched tag for release

## 3.3.0

### Features

- Add `PromptTemplate` to support easy creation of multi-modal prompts

### Bugs

- Fix parsing of optimized prompt returned in a `CompletionResponse`

## 3.2.4

- Make sure `control_factor` gets passed along with `ExplanationRequest`

## 3.2.3

- Make sure model name gets passed along for async batch semnatic embed

## 3.2.2

- Re-relase 3.2.1 again because of deployment issue

## 3.2.1

- Add progress_bar option to batch semantic embedding API
- Add batch_size option to batch semantic embedding API

## 3.2.0

- Add batch_semantic_embed method for processing batches of semantic embeddings

## 3.1.5

- Remove internal search endpoint again (was only accessible for internal users).

## 3.1.4

### Bug fixes

- Do not send max_answers for QA by default, use API's default instead.

## 3.1.3

### Bug fixes

- Add missing import of **PromptGranularity** in _**init**.py_.

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

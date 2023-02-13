import logging
from tokenizers import Tokenizer  # type: ignore
from types import TracebackType
from typing import Any, List, Mapping, Optional, Dict, Sequence, Tuple, Type, Union
import warnings
import aiohttp
from aiohttp import ClientResponse
from aiohttp_retry import RetryClient, ExponentialRetry
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from urllib3.util.retry import Retry

import aleph_alpha_client
from aleph_alpha_client.document import Document
from aleph_alpha_client.explanation import ExplanationRequest, ExplanationResponse
from aleph_alpha_client.image import Image
from aleph_alpha_client.prompt import _to_json, _to_serializable_prompt
from aleph_alpha_client.summarization import SummarizationRequest, SummarizationResponse
from aleph_alpha_client.qa import QaRequest, QaResponse
from aleph_alpha_client.completion import CompletionRequest, CompletionResponse
from aleph_alpha_client.evaluation import EvaluationRequest, EvaluationResponse
from aleph_alpha_client.tokenization import TokenizationRequest, TokenizationResponse
from aleph_alpha_client.detokenization import (
    DetokenizationRequest,
    DetokenizationResponse,
)
from aleph_alpha_client.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
)
from aleph_alpha_client.search import SearchRequest, SearchResponse

POOLING_OPTIONS = ["mean", "max", "last_token", "abs_max"]
RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_REQUEST_TIMEOUT = 305


class QuotaError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BusyError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _raise_for_status(status_code: int, text: str):
    if status_code >= 400:
        if status_code == 400:
            raise ValueError(status_code, text)
        elif status_code == 401:
            raise PermissionError(status_code, text)
        elif status_code == 402:
            raise QuotaError(status_code, text)
        elif status_code == 408:
            raise TimeoutError(status_code, text)
        elif status_code == 503:
            raise BusyError(status_code, text)
        else:
            raise RuntimeError(status_code, text)


class AlephAlphaClient:
    """
    DEPRECATED: :class:`~aleph_alpha_client.AlephAlphaClient` is deprecated and will be removed in the
    next major release. Use :class:`~aleph_alpha_client.Client` or :class:`~aleph_alpha_client.AsyncClient` instead.

    Construct a context object for a specific API and user.

    Parameters:
        host (string, required):
            The hostname of the API host.

        token (string, optional, default None):
            The API token that will be used for authentication.
            This is optional because we also support password authentication.
            If token is None, email and password must be set.

        email (str, optional, default None):
            Email address for authentication at the API. If token is None then email and password will be used for authentication.

        password (str, optional, default None):
            Password for authentication with the email address at the API. If token is None then email and password will be used for authentication.

        request_timeout_seconds (int, optional, default 305):
            Client timeout that will be set for HTTP requests in the `requests` library's API calls.
            Server will close all requests after 300 seconds with an internal server error.

        total_retries(int, optional, default 8)
            The number of retries made in case requests fail with certain retryable status codes. If the last
            retry fails a corresponding exception is raised. Note, that between retries an exponential backoff
            is applied, starting with 0.5 s after the first retry and doubling for each retry made. So with the
            default setting of 8 retries a total wait time of 63.5 s is added between the retries.

    """

    def __init__(
        self,
        host="https://api.aleph-alpha.com",
        token=None,
        email=None,
        password=None,
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT,
        total_retries: int = 8,
    ):
        warnings.warn(
            "AlephAlphaClient is deprecated and will be removed in the next major release. Use Client or AsyncClient instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        if host[-1] != "/":
            host += "/"
        self.host = host

        self.request_timeout_seconds = request_timeout_seconds

        retry_strategy = Retry(
            total=total_retries,
            backoff_factor=0.25,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.requests_session = requests.Session()
        self.requests_session.mount("https://", adapter)
        self.requests_session.mount("http://", adapter)

        # check server version
        expect_release = "1"
        version = self.get_version()
        if not version.startswith(expect_release):
            logging.warning(
                f"Expected API version {expect_release}.x.x, got {version}. Please update client."
            )

        assert token is not None or (email is not None and password is not None)
        self.token = token or self.get_token(email, password)

    def get_version(self):
        response = self.get_request(self.host + "version")
        _raise_for_status(response.status_code, response.text)
        return response.text

    def get_token(self, email, password):
        response = self.post_request(
            self.host + "get_token", json={"email": email, "password": password}
        )
        if response.status_code == 200:
            response_json = response.json()
            return response_json["token"]
        else:
            raise ValueError("cannot get token")

    def get_request(self, url, headers=None):
        return self.requests_session.get(
            url, headers=headers, timeout=self.request_timeout_seconds
        )

    def post_request(self, url, json, params=None, headers=None):
        return self.requests_session.post(
            url,
            headers=headers,
            params=params,
            json=json,
            timeout=self.request_timeout_seconds,
        )

    @property
    def request_headers(self):
        return {
            "Authorization": "Bearer " + self.token,
            "User-Agent": "Aleph-Alpha-Python-Client-" + aleph_alpha_client.__version__,
        }

    def available_models(self):
        """
        Queries all models which are currently available.
        """
        response = self.get_request(
            self.host + "models_available", headers=self.request_headers
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def tokenize(
        self,
        model: str,
        prompt: str,
        tokens: bool = True,
        token_ids: bool = True,
    ):
        """
        Tokenizes the given prompt for the given model.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "tokens": tokens,
            "token_ids": token_ids,
        }

        response = self.post_request(
            self.host + "tokenize",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def detokenize(
        self,
        model: str,
        token_ids: List[int],
    ):
        """
        Detokenizes the given tokens.
        """
        payload: Dict[str, Any] = {"model": model, "token_ids": token_ids}

        response = self.post_request(
            self.host + "detokenize",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def complete(
        self,
        model: str,
        prompt: Union[str, List[Union[str, Image]]] = "",
        hosting: Optional[str] = None,
        maximum_tokens: Optional[int] = 64,
        temperature: Optional[float] = 0.0,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        repetition_penalties_include_prompt: Optional[bool] = False,
        use_multiplicative_presence_penalty: Optional[bool] = False,
        penalty_bias: Optional[str] = None,
        penalty_exceptions: Optional[List[str]] = None,
        penalty_exceptions_include_stop_sequences: Optional[bool] = None,
        best_of: Optional[int] = None,
        n: Optional[int] = 1,
        logit_bias: Optional[Dict[int, float]] = None,
        log_probs: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tokens: Optional[bool] = False,
        disable_optimizations: Optional[bool] = False,
        minimum_tokens: int = 0,
        echo: bool = False,
        use_multiplicative_frequency_penalty: bool = False,
        sequence_penalty: float = 0.0,
        sequence_penalty_min_length: int = 2,
        use_multiplicative_sequence_penalty: bool = False,
        completion_bias_inclusion: Optional[Sequence[str]] = None,
        completion_bias_inclusion_first_token_only: bool = False,
        completion_bias_exclusion: Optional[Sequence[str]] = None,
        completion_bias_exclusion_first_token_only: bool = False,
    ):
        """Generates samples from a prompt.

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            prompt (str, optional, default ""):
                The text to be completed. Unconditional completion can be started with an empty string (default). The prompt may contain a zero shot or few shot task.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.

            maximum_tokens (int, optional, default 64):
                The maximum number of tokens to be generated. Completion will terminate after the maximum number of tokens is reached. Increase this value to generate longer texts. A text is split into tokens. Usually there are more tokens than words. The summed number of tokens of prompt and maximum_tokens depends on the model (for luminous-base, it may not exceed 2048 tokens).

            temperature (float, optional, default 0.0)
                A higher sampling temperature encourages the model to produce less probable outputs ("be more creative"). Values are expected in a range from 0.0 to 1.0. Try high values (e.g. 0.9) for a more "creative" response and the default 0.0 for a well defined and repeatable answer.

                It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

            top_k (int, optional, default 0)
                Introduces random sampling from generated tokens by randomly selecting the next token from the k most likely options. A value larger than 1 encourages the model to be more creative. Set to 0 if repeatable output is to be produced.
                It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

            top_p (float, optional, default 0.0)
                Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p. Set to 0.0 if repeatable output is to be produced.
                It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

            presence_penalty (float, optional, default 0.0)
                The presence penalty reduces the likelihood of generating tokens that are already present in the text. Presence penalty is independent of the number of occurences. Increase the value to produce text that is not repeating the input.

            frequency_penalty (float, optional, default 0.0)
                The frequency penalty reduces the likelihood of generating tokens that are already present in the text. Presence penalty is dependent on the number of occurences of a token.

            repetition_penalties_include_prompt (bool, optional, default False)
                Flag deciding whether presence penalty or frequency penalty are applied to the prompt and completion (True) or only the completion (False)

            use_multiplicative_presence_penalty (bool, optional, default True)
                Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False). This changes the formula stated for presence and frequency penalty.

            penalty_bias (string, optional)
                If set, all tokens in this text will be used in addition to the already penalized tokens for repetition penalties. These consist of the already generated completion tokens and the prompt tokens, if `repetition_penalties_include_prompt` is set to `true`.

                *Potential use case for a chatbot-based completion:*

                Instead of using ``repetition_penalties_include_prompt``\, construct a new string with only the chatbot's reponses included. You would leave out any tokens you use for stop sequences (i.e. ``\\nChatbot:``\), and all user messages.

                With this bias, if you turn up the repetition penalties, you can avoid having your chatbot repeat itself, but not penalize the chatbot from mirroring language provided by the user.

            penalty_exceptions (List(str), optional)
                List of strings that may be generated without penalty, regardless of other penalty settings.

                This is particularly useful for any completion that uses a structured few-shot prompt. For example, if you have a prompt such as:

                ::

                    I want to travel to a location, where I can enjoy both beaches and mountains.

                    - Lake Garda, Italy. This large Italian lake in the southern alps features gravel beaches and mountainside hiking trails.
                    - Mallorca, Spain. This island is famous for its sandy beaches, turquoise water and hilly landscape.
                    - Lake Tahoe, California. This famous lake in the Sierra Nevada mountains offers an amazing variety of outdoor activities.
                    -

                You could set ``penalty_exceptions`` to ``["\\n-"]`` to not penalize the generation of a new list item, but still increase other penalty settings to encourage the generation of new list items without repeating itself.

                By default, we will also include any ``stop_sequences`` you have set, since completion performance can be degraded if expected stop sequences are penalized. You can disable this behavior by settings ``penalty_exceptions_include_stop_sequences`` to ``false``\.

            penalty_exceptions_include_stop_sequences (bool, optional, default true)
                By default, we include any ``stop_sequences`` in ``penalty_exceptions``\, to not penalize the presence of stop sequences that are present in few-shot prompts to provide structure to your completions.

                You can set this to ``false`` if you do not want this behavior.

                See the description of ``penalty_exceptions`` above for more information on what ``penalty_exceptions`` are used for.

            best_of (int, optional, default None)
                Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
                When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.

            n (int, optional, default 1)
                How many completions to generate for each prompt.

            logit_bias (dict mapping token ids to score, optional, default None)
                The logit bias allows to influence the likelihood of generating tokens. A dictionary mapping token ids (int) to a bias (float) can be provided. Such bias is added to the logits as generated by the model.

            log_probs (int, optional, default None)
                Number of top log probabilities to be returned for each generated token. Log probabilities may be used in downstream tasks or to assess the model's certainty when producing tokens.

            stop_sequences (List(str), optional, default None)
                List of strings which will stop generation if they're generated. Stop sequences may be helpful in structured texts.

                Example: In a question answering scenario a text may consist of lines starting with either "Question: " or "Answer: " (alternating). After producing an answer, the model will be likely to generate "Question: ". "Question: " may therfore be used as stop sequence in order not to have the model generate more questions but rather restrict text generation to the answers.

            tokens (bool, optional, default False)
                return tokens of completion

            disable_optimizations (bool, optional, default False)
                We continually research optimal ways to work with our models. By default, we apply these optimizations to both your prompt and  completion for you.

                Our goal is to improve your results while using our API. But you can always pass disable_optimizations: true and we will leave your prompt and completion untouched.

            minimum_tokens (int, default 0)
                Generate at least this number of tokens before an end-of-text token is generated.

            echo (bool, default False)
                Echo the prompt in the completion. This may be especially helpful when log_probs is set to return logprobs for the prompt.

            use_multiplicative_frequency_penalty (bool, default False)
                Flag deciding whether frequency penalty is applied multiplicatively (True) or additively (False).

            sequence_penalty (float, default 0.0)
                Increasing the sequence penalty reduces the likelihood of reproducing token sequences that already appear in the prompt
                (if repetition_penalties_include_prompt is True) and prior completion.

            sequence_penalty_min_length (int, default 2)
                Minimal number of tokens to be considered as sequence. Must be greater or eqaul 2.

            use_multiplicative_sequence_penalty (bool, default False)
                Flag deciding whether sequence penalty is applied multiplicatively (True) or additively (False).

            completion_bias_inclusion (List[str], default [])
                Bias the completion to only generate options within this list;
                all other tokens are disregarded at sampling

            completion_bias_inclusion_first_token_only (bool, default False)
                Only consider the first token for the completion_bias_inclusion

            completion_bias_exclusion (List[str], default [])
                Bias the completion to NOT generate options within this list;
                all other tokens are unaffected in sampling

            completion_bias_exclusion_first_token_only (bool, default False)
                Only consider the first token for the completion_bias_exclusion
        """

        payload = {
            "model": model,
            "prompt": _to_serializable_prompt(prompt=prompt),
            "maximum_tokens": maximum_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "penalty_bias": penalty_bias,
            "penalty_exceptions": penalty_exceptions,
            "penalty_exceptions_include_stop_sequences": penalty_exceptions_include_stop_sequences,
            "best_of": best_of,
            "n": n,
            "logit_bias": logit_bias,
            "log_probs": log_probs,
            "repetition_penalties_include_prompt": repetition_penalties_include_prompt,
            "use_multiplicative_presence_penalty": use_multiplicative_presence_penalty,
            "stop_sequences": stop_sequences,
            "tokens": tokens,
            "disable_optimizations": disable_optimizations,
            "minimum_tokens": minimum_tokens,
            "echo": echo,
            "use_multiplicative_frequency_penalty": use_multiplicative_frequency_penalty,
            "sequence_penalty": sequence_penalty,
            "sequence_penalty_min_length": sequence_penalty_min_length,
            "use_multiplicative_sequence_penalty": use_multiplicative_sequence_penalty,
            "completion_bias_inclusion": completion_bias_inclusion or [],
            "completion_bias_inclusion_first_token_only": completion_bias_inclusion_first_token_only,
            "completion_bias_exclusion": completion_bias_exclusion or [],
            "completion_bias_exclusion_first_token_only": completion_bias_exclusion_first_token_only,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "complete",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        response_json = response.json()
        if response_json.get("optimized_prompt") is not None:
            # Return a message to the user that we optimized their prompt
            print(
                'We optimized your prompt before sending it to the model. The optimized prompt is available at result["optimized_prompt"]. If you do not want these optimizations applied, you can pass the disable_optimizations flag to your request.'
            )
        return response_json

    def embed(
        self,
        model: str,
        prompt: Union[str, Sequence[Union[str, Image]]],
        pooling: List[str],
        layers: List[int],
        hosting: Optional[str] = None,
        tokens: Optional[bool] = False,
        type: Optional[str] = None,
        normalize: bool = False,
    ):
        """
        Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            prompt (str, required):
               The text to be embedded.

            layers (List[int], required):
               A list of layer indices from which to return embeddings.

               * Index 0 corresponds to the word embeddings used as input to the first transformer layer
               * Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
               * Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last layer etc.

            pooling (List[str])
                Pooling operation to use.
                Pooling operations include:

                * mean: aggregate token embeddings across the sequence dimension using an average
                * max: aggregate token embeddings across the sequence dimension using a maximum
                * last_token: just use the last token
                * abs_max: aggregate token embeddings across the sequence dimension using a maximum of absolute values

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.

            tokens (bool, optional, default False)
                Flag indicating whether the tokenized prompt is to be returned (True) or not (False)

            type
                Type of the embedding (e.g. symmetric or asymmetric)

            normalize
                Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.
        """

        serializable_prompt = _to_serializable_prompt(
            prompt=prompt, at_least_one_token=True
        )

        if tokens is None:
            tokens = False

        payload = {
            "model": model,
            "prompt": serializable_prompt,
            "layers": layers,
            "tokens": tokens,
            "pooling": pooling,
            "type": type,
            "normalize": normalize,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "embed",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def semantic_embed(
        self,
        model: str,
        request: SemanticEmbeddingRequest,
        hosting: Optional[str] = None,
        normalize: bool = False,
    ):
        """
        Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.

            request (SemanticEmbeddingRequest, required)
                NamedTuple containing all necessary request parameters.

            normalize
                Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.
        """

        serializable_prompt = request.prompt.to_json()

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": serializable_prompt,
            "representation": request.representation.value,
            "compress_to_size": request.compress_to_size,
            "normalize": normalize,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "semantic_embed",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def evaluate(
        self,
        model: str,
        completion_expected,
        hosting: Optional[str] = None,
        prompt: Union[str, List[Union[str, Image]]] = "",
    ):
        """
        Evaluates the model's likelihood to produce a completion given a prompt.

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            completion_expected (str, required):
                The ground truth completion expected to be produced given the prompt.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.

            prompt (str, optional, default ""):
                The text to be completed. Unconditional completion can be used with an empty string (default). The prompt may contain a zero shot or few shot task.
        """

        serializable_prompt = _to_serializable_prompt(prompt=prompt)

        payload = {
            "model": model,
            "prompt": serializable_prompt,
            "completion_expected": completion_expected,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "evaluate",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def qa(
        self,
        model: str,
        query: str,
        documents: List[Document],
        maximum_tokens: int = 64,
        max_chunk_size: int = 175,
        disable_optimizations: bool = False,
        max_answers: int = 0,
        min_score: float = 0.0,
        hosting: Optional[str] = None,
    ):
        """
        Answers a question about documents.

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            query (str, required):
                The question to be answered about the documents by the model.

            documents (List[Document], required):
                A list of documents. This can be either docx documents or text/image prompts.

            maximum_tokens (int, default 64):
                The maximum number of tokens to be generated. Completion will terminate after the maximum number of tokens is reached.

                Increase this value to generate longer texts. A text is split into tokens. Usually there are more tokens than words. The summed number of tokens of prompt and maximum_tokens depends on the model (for luminous-base, it may not exceed 2048 tokens).

            max_chunk_size (int, default 175):
                Long documents will be split into chunks if they exceed max_chunk_size.
                The splitting will be done along the following boundaries until all chunks are shorter than max_chunk_size or all splitting criteria have been exhausted.
                The splitting boundaries are, in the given order:
                1. Split first by double newline
                (assumed to mark the boundary between 2 paragraphs).
                2. Split paragraphs that are still too long by their median sentence as long as we can still find multiple sentences in the paragraph.
                3. Split each remaining chunk of a paragraph or sentence further along white spaces until each chunk is smaller than max_chunk_size or until no whitespace can be found anymore.

            disable_optimizations  (bool, default False)
                We continually research optimal ways to work with our models. By default, we apply these optimizations to both your query, documents, and answers for you.
                Our goal is to improve your results while using our API. But you can always pass `disable_optimizations: true` and we will leave your query, documents, and answers untouched.

            max_answers (int, default 0):
                The upper limit of maximum number of answers.

            min_score (float, default 0.0):
                The lower limit of minimum score for every answer.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.
        """

        payload = {
            "model": model,
            "query": query,
            "documents": [
                document._to_serializable_document() for document in documents
            ],
            "maximum_tokens": maximum_tokens,
            "max_answers": max_answers,
            "min_score": min_score,
            "max_chunk_size": max_chunk_size,
            "disable_optimizations": disable_optimizations,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "qa",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def summarize(
        self,
        model: str,
        request: SummarizationRequest,
        hosting: Optional[str] = None,
    ):
        """
        Summarizes a document.

        Parameters:
            model (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.

            request (SemanticEmbeddingRequest, required)
                NamedTuple containing all necessary request parameters.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "document": request.document._to_serializable_document(),
            "disable_optimizations": request.disable_optimizations,
        }

        if hosting is not None:
            payload["hosting"] = hosting

        response = self.post_request(
            self.host + "summarize",
            headers=self.request_headers,
            json=payload,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()

    def _explain(
        self,
        model: str,
        request: ExplanationRequest,
        hosting: Optional[str] = None,
    ):
        body = {
            "model": model,
            "prompt": [_to_json(item) for item in request.prompt.items],
            "target": request.target,
            "suppression_factor": request.suppression_factor,
            "conceptual_suppression_threshold": request.conceptual_suppression_threshold,
            "normalize": request.normalize,
            "square_outputs": request.square_outputs,
            "prompt_explain_indices": request.prompt_explain_indices,
        }

        if hosting is not None:
            body["hosting"] = hosting

        response = self.post_request(
            f"{self.host}explain",
            headers=self.request_headers,
            json=body,
        )
        _raise_for_status(response.status_code, response.text)
        return response.json()


AnyRequest = Union[
    CompletionRequest,
    EmbeddingRequest,
    EvaluationRequest,
    TokenizationRequest,
    DetokenizationRequest,
    SemanticEmbeddingRequest,
    QaRequest,
    SummarizationRequest,
    ExplanationRequest,
    SearchRequest,
]


class Client:
    """
    Construct a client for synchronous requests given a user token

    Parameters:
        token (string, required):
            The API token that will be used for authentication.

        host (string, required, default "https://api.aleph-alpha.com"):
            The hostname of the API host.

        hosting(string, optional, default None):
            Determines in which datacenters the request may be processed.
            You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

            Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
            own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

            Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
            Choose this option for maximal data privacy.

        request_timeout_seconds (int, optional, default 305):
            Client timeout that will be set for HTTP requests in the `requests` library's API calls.
            Server will close all requests after 300 seconds with an internal server error.

        total_retries(int, optional, default 8)
            The number of retries made in case requests fail with certain retryable status codes. If the last
            retry fails a corresponding exception is raised. Note, that between retries an exponential backoff
            is applied, starting with 0.5 s after the first retry and doubling for each retry made. So with the
            default setting of 8 retries a total wait time of 63.5 s is added between the retries.

        nice(bool, required, default False):
            Setting this to True, will signal to the API that you intend to be nice to other users
            by de-prioritizing your request below concurrent ones.

    Example usage:
        >>> request = CompletionRequest(
                prompt=Prompt.from_text(f"Request"), maximum_tokens=64
            )
        >>> client = Client(token=os.environ["AA_TOKEN"])
        >>> response: CompletionResponse = client.complete(request, "luminous-base")
    """

    def __init__(
        self,
        token: str,
        host: str = "https://api.aleph-alpha.com",
        hosting: Optional[str] = None,
        request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        total_retries: int = 8,
        nice: bool = False,
    ) -> None:

        if host[-1] != "/":
            host += "/"
        self.host = host
        self.hosting = hosting
        self.request_timeout_seconds = request_timeout_seconds
        self.token = token
        self.nice = nice

        retry_strategy = Retry(
            total=total_retries,
            backoff_factor=0.25,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.headers = CaseInsensitiveDict(
            {
                "Authorization": "Bearer " + self.token,
                "User-Agent": "Aleph-Alpha-Python-Client-"
                + aleph_alpha_client.__version__,
            }
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_version(self) -> str:
        """Gets version of the AlephAlpha HTTP API."""
        return self._get_request("version").text

    def _get_request(self, endpoint: str) -> Response:
        response = self.session.get(self.host + endpoint)
        if not response.ok:
            _raise_for_status(response.status_code, response.text)
        return response

    def _post_request(
        self,
        endpoint: str,
        request: AnyRequest,
        model: Optional[str],
    ) -> Dict[str, Any]:

        json_body = self._build_json_body(request, model)

        query_params = self._build_query_parameters()

        response = self.session.post(
            self.host + endpoint,
            json=json_body,
            params=query_params,
            timeout=self.request_timeout_seconds,
        )
        if not response.ok:
            _raise_for_status(response.status_code, response.text)
        return response.json()

    def _build_query_parameters(self) -> Mapping[str, str]:
        return {
            # Cannot use str() here because we want lowercase true/false in query string.
            # Also do not want to send the nice flag with every request if it is false
            **({"nice": "true"} if self.nice else {}),
        }

    def _build_json_body(
        self, request: AnyRequest, model: Optional[str]
    ) -> Mapping[str, Any]:
        json_body = request.to_json()

        if model is not None:
            json_body["model"] = model
        if self.hosting is not None:
            json_body["hosting"] = self.hosting
        return json_body

    def models(self) -> List[Mapping[str, Any]]:
        """
        Queries all models which are currently available.

        For documentation of the response, see https://docs.aleph-alpha.com/api/available-models/
        """
        response = self._get_request("models_available")
        return response.json()

    def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        """Generates completions given a prompt.

        Parameters:
            request (CompletionRequest, required):
                Parameters for the requested completion.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # create a prompt
            >>> prompt = Prompt.from_text("An apple a day, ")
            >>>
            >>> # create a completion request
            >>> request = CompletionRequest(
                    prompt=prompt,
                    maximum_tokens=32,
                    stop_sequences=["###","\\n"],
                    temperature=0.12
                )
            >>>
            >>> # complete the prompt
            >>> result = client.complete(request, model=model_name)
        """
        response = self._post_request("complete", request, model)
        return CompletionResponse.from_json(response)

    def tokenize(
        self,
        request: TokenizationRequest,
        model: str,
    ) -> TokenizationResponse:
        """Tokenizes the given prompt for the given model.

        Parameters:
            request (TokenizationRequest, required):
                Parameters for the requested tokenization.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = TokenizationRequest(
                    prompt="hello", token_ids=True, tokens=True
                )
            >>> response = client.tokenize(request, model=model_name)
        """
        response = self._post_request(
            "tokenize",
            request,
            model,
        )
        return TokenizationResponse.from_json(response)

    def detokenize(
        self,
        request: DetokenizationRequest,
        model: str,
    ) -> DetokenizationResponse:
        """Detokenizes the given prompt for the given model.

        Parameters:
            request (DetokenizationRequest, required):
                Parameters for the requested detokenization.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = DetokenizationRequest(token_ids=[2, 3, 4])
            >>> response = client.detokenize(request, model=model_name)
        """
        response = self._post_request(
            "detokenize",
            request,
            model,
        )
        return DetokenizationResponse.from_json(response)

    def embed(
        self,
        request: EmbeddingRequest,
        model: str,
    ) -> EmbeddingResponse:
        """Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            request (EmbeddingRequest, required):
                Parameters for the requested embedding.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = EmbeddingRequest(prompt=Prompt.from_text(
                    "This is an example."), layers=[-1], pooling=["mean"]
                )
            >>> result = client.embed(request, model=model_name)
        """
        response = self._post_request(
            "embed",
            request,
            model,
        )
        return EmbeddingResponse.from_json(response)

    def semantic_embed(
        self,
        request: SemanticEmbeddingRequest,
        model: str,
    ) -> SemanticEmbeddingResponse:
        """Embeds a text and returns vectors that can be used for downstream tasks
        (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            request (SemanticEmbeddingRequest, required):
                Parameters for the requested semnatic embedding.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # function for symmetric embedding
            >>> def embed_symmetric(text: str):
                    # Create an embeddingrequest with the type set to symmetric
                    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(
                        text), representation=SemanticRepresentation.Symmetric)
                    # create the embedding
                    result = client.semantic_embed(request, model=model_name)
                    return result.embedding
            >>>
            >>> # function to calculate similarity
            >>> def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
                    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
                    sumxx, sumxy, sumyy = 0, 0, 0
                    for i in range(len(v1)):
                        x = v1[i]; y = v2[i]
                        sumxx += x*x
                        sumyy += y*y
                        sumxy += x*y
                    return sumxy/math.sqrt(sumxx*sumyy)
            >>>
            >>> # define the texts
            >>> text_a = "The sun is shining"
            >>> text_b = "Il sole splende"
            >>>
            >>> # show the similarity
            >>> print(cosine_similarity(embed_symmetric(text_a), embed_symmetric(text_b)))
        """
        response = self._post_request(
            "semantic_embed",
            request,
            model,
        )
        return SemanticEmbeddingResponse.from_json(response)

    def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        """Evaluates the model's likelihood to produce a completion given a prompt.

        Parameters:
            request (EvaluationRequest, required):
                Parameters for the requested evaluation.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = EvaluationRequest(
                    prompt=Prompt.from_text("hello"), completion_expected=" world"
                )
            >>> response = client.evaluate(request, model=model_name)
        """
        response = self._post_request(
            "evaluate",
            request,
            model,
        )
        return EvaluationResponse.from_json(response)

    def qa(
        self,
        request: QaRequest,
        model: Optional[str] = None,
        beta: bool = False,
    ) -> QaResponse:
        """Answers a question about documents.

        Parameters:
            request (QaRequest, required):
                Parameters for the qa request.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

            beta (bool, optional, default False):
                Opt-in use of new beta implementations of the QA endpoint, if available. Setting this to true will only change the
                behavior if there is a currently running beta available.

                Disclaimer: Pricing, performance, and behavior are subject to change at any time for beta implementations. Please be
                aware of this before using. You can always reach out to support@alepha-alpha.com if you have questions.

        Examples:
            >>> request = QaRequest(
                    query="Who likes pizza?",
                    documents=[Document.from_text("Andreas likes pizza.")],
                )
            >>> response = client.qa(request, model="luminous-extended")
        """
        response = self._post_request(
            "qa/beta" if beta else "qa",
            request,
            model,
        )
        return QaResponse.from_json(response)

    def summarize(
        self,
        request: SummarizationRequest,
        model: str,
    ) -> SummarizationResponse:
        """Summarizes a document.

        Parameters:
            request (SummarizationRequest, required):
                Parameters for the requested summarization.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = SummarizationRequest(
                    document=Document.from_text("Andreas likes pizza."),
                )
            >>> response = client.summarize(request, model="luminous-extended")
        """
        response = self._post_request(
            "summarize",
            request,
            model,
        )
        return SummarizationResponse.from_json(response)

    def _explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        response = self._post_request(
            "explain",
            request,
            model,
        )
        return ExplanationResponse.from_json(response)

    def _search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        """
        For details see https://www.aleph-alpha.com/luminous-explore-a-model-for-world-class-semantic-representation
        """
        response = self._post_request("search", request, None)
        return SearchResponse.from_json(response)

    def tokenizer(self, model: str) -> Tokenizer:
        """Returns a Tokenizer instance with the settings that were used to train the model.

        Examples:
            >>> tokenizer = client.tokenizer(model="luminous-extended")
            >>> tokenized_prompt = tokenizer.encode("Hello world")
        """
        return Tokenizer.from_str(self._get_request(f"models/{model}/tokenizer").text)


class AsyncClient:
    """
    Construct a context object for asynchronous requests given a user token

    Parameters:
        token (string, required):
            The API token that will be used for authentication.

        host (string, required, default "https://api.aleph-alpha.com"):
            The hostname of the API host.

        hosting(string, optional, default None):
            Determines in which datacenters the request may be processed.
            You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

            Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
            own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

            Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
            Choose this option for maximal data privacy.

        request_timeout_seconds (int, optional, default 305):
            Client timeout that will be set for HTTP requests in the `aiohttp` library's API calls.
            Server will close all requests after 300 seconds with an internal server error.

        total_retries(int, optional, default 8)
            The number of retries made in case requests fail with certain retryable status codes. If the last
            retry fails a corresponding exception is raised. Note, that between retries an exponential backoff
            is applied, starting with 0.25 s after the first request and doubling for each retry made. So with the
            default setting of 8 retries a total wait time of 63.75 s is added between the retries.

        nice(bool, required, default False):
            Setting this to True, will signal to the API that you intend to be nice to other users
            by de-prioritizing your request below concurrent ones.

    Example usage:
        >>> request = CompletionRequest(prompt=Prompt.from_text(f"Request"), maximum_tokens=64)
        >>> async with AsyncClient(token=os.environ["AA_TOKEN"]) as client:
                response: CompletionResponse = await client.complete(request, "luminous-base")
    """

    def __init__(
        self,
        token: str,
        host: str = "https://api.aleph-alpha.com",
        hosting: Optional[str] = None,
        request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        total_retries: int = 8,
        nice: bool = False,
    ) -> None:
        if host[-1] != "/":
            host += "/"
        self.host = host
        self.hosting = hosting
        self.request_timeout_seconds = request_timeout_seconds
        self.token = token
        self.nice = nice

        retry_options = ExponentialRetry(
            attempts=total_retries + 1,
            start_timeout=0.25,
            statuses=set(RETRY_STATUS_CODES),
        )
        self.session = RetryClient(
            raise_for_status=False,
            retry_options=retry_options,
            timeout=aiohttp.ClientTimeout(self.request_timeout_seconds),
            headers={
                "Authorization": "Bearer " + self.token,
                "User-Agent": "Aleph-Alpha-Python-Client-"
                + aleph_alpha_client.__version__,
            },
        )

    async def close(self):
        """Needs to be called at end of lifetime if the AsyncClient object is not used as a context manager."""
        await self.session.close()

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "AsyncClient":
        await self.session.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        await self.session.__aexit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    async def get_version(self) -> str:
        """Gets version of the AlephAlpha HTTP API."""
        return await self._get_request_text("version")

    async def _get_request_text(self, endpoint: str) -> str:
        async with self.session.get(
            self.host + endpoint,
        ) as response:
            if not response.ok:
                _raise_for_status(response.status, await response.text())
            return await response.text()

    async def _get_request_json(
        self, endpoint: str
    ) -> Union[List[Mapping[str, Any]], Mapping[str, Any]]:
        async with self.session.get(
            self.host + endpoint,
        ) as response:
            if not response.ok:
                _raise_for_status(response.status, await response.text())
            return await response.json()

    async def _post_request(
        self,
        endpoint: str,
        request: AnyRequest,
        model: Optional[str],
    ) -> Dict[str, Any]:

        json_body = self._build_json_body(request, model)

        query_params = self._build_query_parameters()

        async with self.session.post(
            self.host + endpoint, json=json_body, params=query_params
        ) as response:
            if not response.ok:
                _raise_for_status(response.status, await response.text())
            return await response.json()

    def _build_query_parameters(self) -> Mapping[str, str]:
        return {
            # cannot use str() here because we want lowercase true/false in query string
            # Also do not want to send the nice flag with every request if it is false
            **({"nice": "true"} if self.nice else {}),
        }

    def _build_json_body(
        self, request: AnyRequest, model: Optional[str]
    ) -> Mapping[str, Any]:
        json_body = request.to_json()

        if model is not None:
            json_body["model"] = model
        if self.hosting is not None:
            json_body["hosting"] = self.hosting
        return json_body

    async def models(self) -> List[Mapping[str, Any]]:
        """
        Queries all models which are currently available.

        For documentation of the response, see https://docs.aleph-alpha.com/api/available-models/
        """
        return await self._get_request_json("models_available")  # type: ignore

    async def complete(
        self,
        request: CompletionRequest,
        model: str,
    ) -> CompletionResponse:
        """Generates completions given a prompt.

        Parameters:
            request (CompletionRequest, required):
                Parameters for the requested completion.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # create a prompt
            >>> prompt = Prompt.from_text("An apple a day, ")
            >>>
            >>> # create a completion request
            >>> request = CompletionRequest(
                    prompt=prompt,
                    maximum_tokens=32,
                    stop_sequences=["###","\\n"],
                    temperature=0.12
                )
            >>>
            >>> # complete the prompt
            >>> result = await client.complete(request, model=model_name)
        """
        response = await self._post_request(
            "complete",
            request,
            model,
        )
        return CompletionResponse.from_json(response)

    async def tokenize(
        self,
        request: TokenizationRequest,
        model: str,
    ) -> TokenizationResponse:
        """Tokenizes the given prompt for the given model.

        Parameters:
            request (TokenizationRequest, required):
                Parameters for the requested tokenization.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)
            >>> response = await client.tokenize(request, model=model_name)
        """
        response = await self._post_request(
            "tokenize",
            request,
            model,
        )
        return TokenizationResponse.from_json(response)

    async def detokenize(
        self,
        request: DetokenizationRequest,
        model: str,
    ) -> DetokenizationResponse:
        """Detokenizes the given prompt for the given model.

        Parameters:
            request (DetokenizationRequest, required):
                Parameters for the requested detokenization.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = DetokenizationRequest(token_ids=[2, 3, 4])
            >>> response = await client.detokenize(request, model=model_name)
        """
        response = await self._post_request(
            "detokenize",
            request,
            model,
        )
        return DetokenizationResponse.from_json(response)

    async def embed(
        self,
        request: EmbeddingRequest,
        model: str,
    ) -> EmbeddingResponse:
        """Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            request (EmbeddingRequest, required):
                Parameters for the requested embedding.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = EmbeddingRequest(prompt=Prompt.from_text("This is an example."), layers=[-1], pooling=["mean"])
            >>> result = await client.embed(request, model=model_name)
        """
        response = await self._post_request(
            "embed",
            request,
            model,
        )
        return EmbeddingResponse.from_json(response)

    async def semantic_embed(
        self,
        request: SemanticEmbeddingRequest,
        model: str,
    ) -> SemanticEmbeddingResponse:
        """Embeds a text and returns vectors that can be used for downstream tasks
        (e.g. semantic similarity) and models (e.g. classifiers).

        Parameters:
            request (SemanticEmbeddingRequest, required):
                Parameters for the requested semnatic embedding.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # function for symmetric embedding
            >>> async def embed_symmetric(text: str):
                    # Create an embeddingrequest with the type set to symmetric
                    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), representation=SemanticRepresentation.Symmetric)
                    # create the embedding
                    result = await client.semantic_embed(request, model=model_name)
                    return result.embedding
            >>>
            >>> # function to calculate similarity
            >>> def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
                    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
                    sumxx, sumxy, sumyy = 0, 0, 0
                    for i in range(len(v1)):
                        x = v1[i]; y = v2[i]
                        sumxx += x*x
                        sumyy += y*y
                        sumxy += x*y
                    return sumxy/math.sqrt(sumxx*sumyy)
            >>>
            >>> # define the texts
            >>> text_a = "The sun is shining"
            >>> text_b = "Il sole splende"
            >>>
            >>> # show the similarity
            >>> print(cosine_similarity(await embed_symmetric(text_a), await embed_symmetric(text_b)))
        """
        response = await self._post_request(
            "semantic_embed",
            request,
            model,
        )
        return SemanticEmbeddingResponse.from_json(response)

    async def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        """Evaluates the model's likelihood to produce a completion given a prompt.

        Parameters:
            request (EvaluationRequest, required):
                Parameters for the requested evaluation.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = EvaluationRequest(
                    prompt=Prompt.from_text("hello"), completion_expected=" world"
                )
            >>> response = await client.evaluate(request, model=model_name)
        """
        response = await self._post_request(
            "evaluate",
            request,
            model,
        )
        return EvaluationResponse.from_json(response)

    async def qa(
        self,
        request: QaRequest,
        model: Optional[str] = None,
        beta: bool = False,
    ) -> QaResponse:
        """Answers a question about documents.

        Parameters:
            request (QaRequest, required):
                Parameters for the qa request.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

            beta (bool, optional, default False):
                Opt-in use of new beta implementations of the QA endpoint, if available. Setting this to true will only change the
                behavior if there is a currently running beta available.

                Disclaimer: Pricing, performance, and behavior are subject to change at any time for beta implementations. Please be
                aware of this before using. You can always reach out to support@alepha-alpha.com if you have questions.

        Examples:
            >>> request = QaRequest(
                    query="Who likes pizza?",
                    documents=[Document.from_text("Andreas likes pizza.")],
                )
            >>> response = await client.qa(request, model="luminous-extended")
        """
        response = await self._post_request(
            "qa/beta" if beta else "qa",
            request,
            model,
        )
        return QaResponse.from_json(response)

    async def summarize(
        self,
        request: SummarizationRequest,
        model: str,
    ) -> SummarizationResponse:
        """Summarizes a document.

        Parameters:
            request (SummarizationRequest, required):
                Parameters for the requested summarization.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = SummarizationRequest(
                    document=Document.from_text("Andreas likes pizza."),
                )
            >>> response = await client.summarize(request, model="luminous-extended")
        """
        response = await self._post_request(
            "summarize",
            request,
            model,
        )
        return SummarizationResponse.from_json(response)

    async def _explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        response = await self._post_request(
            "explain",
            request,
            model,
        )
        return ExplanationResponse.from_json(response)

    async def _search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        """
        For details see https://www.aleph-alpha.com/luminous-explore-a-model-for-world-class-semantic-representation
        """
        response = await self._post_request("search", request, None)
        return SearchResponse.from_json(response)

    async def tokenizer(self, model: str) -> Tokenizer:
        """Returns a Tokenizer instance with the settings that were used to train the model.

        Examples:
            >>> tokenizer = await client.tokenizer(model="luminous-extended")
            >>> tokenized_prompt = tokenizer.encode("Hello world")
        """
        response = await self._get_request_text(f"models/{model}/tokenizer")
        return Tokenizer.from_str(response)

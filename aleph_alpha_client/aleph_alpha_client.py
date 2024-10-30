import json
import warnings

from packaging import version
from tokenizers import Tokenizer  # type: ignore
from types import TracebackType
from typing import (
    Any,
    AsyncGenerator,
    List,
    Mapping,
    Optional,
    Dict,
    Sequence,
    Type,
    Union,
)
import aiohttp
import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from urllib3.util.retry import Retry
from tqdm.asyncio import tqdm

import aleph_alpha_client
from aleph_alpha_client.explanation import (
    ExplanationRequest,
    ExplanationResponse,
)
from aleph_alpha_client.summarization import SummarizationRequest, SummarizationResponse
from aleph_alpha_client.qa import QaRequest, QaResponse
from aleph_alpha_client.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamItem,
    stream_item_from_json,
)
from aleph_alpha_client.chat import ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamChunk, Usage, stream_chat_item_from_json
from aleph_alpha_client.evaluation import EvaluationRequest, EvaluationResponse
from aleph_alpha_client.tokenization import TokenizationRequest, TokenizationResponse
from aleph_alpha_client.detokenization import (
    DetokenizationRequest,
    DetokenizationResponse,
)
from aleph_alpha_client.embedding import (
    BatchSemanticEmbeddingRequest,
    BatchSemanticEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
)
from aleph_alpha_client.version import MIN_API_VERSION

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


def _check_api_version(version_str: str):
    api_ver = version.parse(MIN_API_VERSION)
    ver = version.parse(version_str)
    valid = api_ver.major == ver.major and api_ver <= ver
    if not valid:
        raise RuntimeError(
            f"The aleph alpha client requires at least api version {api_ver}, found version {ver}"
        )


AnyRequest = Union[
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
    EvaluationRequest,
    TokenizationRequest,
    DetokenizationRequest,
    SemanticEmbeddingRequest,
    BatchSemanticEmbeddingRequest,
    QaRequest,
    SummarizationRequest,
    ExplanationRequest,
    ExplanationRequest,
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

        verify_ssl(bool, optional, default True)
            Setting this to False will disable checking for SSL when doing requests.

        tags(Optional[Sequence[str]], optional, default None)
            Internal feature.

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
        verify_ssl=True,
        tags: Optional[Sequence[str]] = None,
        pool_size: int = 10,
    ) -> None:
        if host[-1] != "/":
            host += "/"
        self.host = host
        self.hosting = hosting
        self.request_timeout_seconds = request_timeout_seconds
        self.token = token
        self.nice = nice
        self.tags = tags

        retry_strategy = Retry(
            total=total_retries,
            backoff_factor=0.25,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.session.headers = CaseInsensitiveDict(
            {
                "Authorization": "Bearer " + self.token,
                "User-Agent": "Aleph-Alpha-Python-Client-"
                + aleph_alpha_client.__version__,
            }
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def validate_version(self) -> None:
        """Gets version of the AlephAlpha HTTP API."""
        _check_api_version(self.get_version())

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
        model: Optional[str] = None,
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
        json_body = dict(request.to_json())

        if model is not None:
            json_body["model"] = model
        if self.hosting is not None:
            json_body["hosting"] = self.hosting
        if self.tags is not None:
            json_body["tags"] = self.tags
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

    def chat(
        self,
        request: ChatRequest,
        model: str,
    ) -> ChatResponse:
        """Chat with a model.

        Parameters:
            request (ChatRequest, required):
                Parameters for the requested chat.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # create a chat request
            >>> request = ChatRequest(
                    messages=[Message(role="user", content="Hello, how are you?")],
                    model=model,
                )
            >>>
            >>> # chat with the model
            >>> result = client.chat(request, model=model_name)
        """
        response = self._post_request("chat/completions", request, model)
        return ChatResponse.from_json(response)

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
                Parameters for the requested semantic embedding.

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

    def batch_semantic_embed(
        self,
        request: BatchSemanticEmbeddingRequest,
        model: Optional[str] = None,
    ) -> BatchSemanticEmbeddingResponse:
        """Embeds a sequence of texts or images and returns vectors in the same order as they
        were provided. If more than 100 prompts are provided then this method will chunk them
        into batches of 100 prompts that will be sent to the API.

        Parameters:
            request (BatchSemanticEmbeddingRequest, required):
                Parameters for the requested semantic embeddings.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # function for symmetric embedding
            >>> def embed_symmetric(texts: Sequence[str]):
                    # Create an embeddingrequest with the type set to symmetric
                    request = BatchSemanticEmbeddingRequest(
                        prompts=[Prompt.from_text(text) for text in texts],
                        representation=SemanticRepresentation.Symmetric
                    )
                    # create the embedding
                    result = client.batch_semantic_embed(request, model=model_name)
                    return result.embedding
        """

        responses: List[EmbeddingVector] = []
        model_version = ""
        num_tokens_prompt_total = 0
        # The API currently only supports batch semantic embedding requests with up to 100
        # prompts per batch. As a convenience for users, this function chunks larger requests.
        for batch_request in _generate_semantic_embedding_batches(request):
            raw_response = self._post_request(
                "batch_semantic_embed",
                batch_request,
                model,
            )
            response = BatchSemanticEmbeddingResponse.from_json(raw_response)
            model_version = response.model_version
            responses.extend(response.embeddings)
            num_tokens_prompt_total += response.num_tokens_prompt_total

        return BatchSemanticEmbeddingResponse(
            model_version=model_version,
            embeddings=responses,
            num_tokens_prompt_total=num_tokens_prompt_total,
        )

    def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        """Evaluates the model's likelihood to produce a completion given a prompt.

        Parameters:
            request (EvaluationRequest, required):
                Parameters for the requested evaluation.

            model (string, required):
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

    def qa(self, request: QaRequest) -> QaResponse:
        """DEPRECATED: `qa` is deprecated and will be removed in the next major release. New
        methods of processing Q&A tasks will be provided before this is removed.

        Answers a question about documents.

        Parameters:
            request (QaRequest, required):
                Parameters for the qa request.

        Examples:
            >>> request = QaRequest(
                    query="Who likes pizza?",
                    documents=[Document.from_text("Andreas likes pizza.")],
                )
            >>> response = client.qa(request)
        """
        warnings.warn(
            "qa is deprecated and will be removed in the next major release. New methods of processing Q&A tasks will be provided before this is removed.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        response = self._post_request("qa", request)
        return QaResponse.from_json(response)

    def summarize(
        self,
        request: SummarizationRequest,
    ) -> SummarizationResponse:
        """DEPRECATED: `summarize` is deprecated and will be removed in the next major release. New
        methods of processing Summarization tasks will be provided before this is removed.

        Summarizes a document.

        Parameters:
            request (SummarizationRequest, required):
                Parameters for the requested summarization.

        Examples:
            >>> request = SummarizationRequest(
                    document=Document.from_text("Andreas likes pizza."),
                )
            >>> response = client.summarize(request, model="luminous-extended")
        """
        warnings.warn(
            "summarize is deprecated and will be removed in the next major release. New methods of processing Q&A tasks will be provided before this is removed.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        response = self._post_request(
            "summarize",
            request,
        )
        return SummarizationResponse.from_json(response)

    def explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        """Better understand the source of a completion, specifically on how much each section of a
        prompt impacts each token of the completion.

        Parameters:
            request (ExplanationRequest, required):
                Parameters for the requested explanation.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = ExplanationRequest(
                    prompt=Prompt.from_text("Andreas likes"),
                    target=" pizza."
                )
            >>> response = client.explain(request, model="luminous-extended")
        """
        response = self._post_request(
            "explain",
            request,
            model,
        )
        return ExplanationResponse.from_json(response)

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

        verify_ssl(bool, optional, default True)
            Setting this to False will disable checking for SSL when doing requests.

        tags(Optional[Sequence[str]], optional, default None)
            Internal feature.

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
        verify_ssl=True,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        if host[-1] != "/":
            host += "/"
        self.host = host
        self.hosting = hosting
        self.request_timeout_seconds = request_timeout_seconds
        self.token = token
        self.nice = nice
        self.tags = tags

        retry_options = ExponentialRetry(
            attempts=total_retries + 1,
            exceptions={aiohttp.ClientConnectionError},
            start_timeout=0.25,
            statuses=set(RETRY_STATUS_CODES),
        )
        connector = aiohttp.TCPConnector(verify_ssl=verify_ssl)
        self.session = RetryClient(
            trust_env=True,  # same behaviour as requests/(Sync)Client wrt. http_proxy
            raise_for_status=False,
            retry_options=retry_options,
            timeout=aiohttp.ClientTimeout(self.request_timeout_seconds),
            headers={
                "Authorization": "Bearer " + self.token,
                "User-Agent": "Aleph-Alpha-Python-Client-"
                + aleph_alpha_client.__version__,
            },
            connector=connector,
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

    async def validate_version(self) -> None:
        _check_api_version(await self.get_version())

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
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        json_body = self._build_json_body(request, model)

        query_params = self._build_query_parameters()

        async with self.session.post(
            self.host + endpoint, json=json_body, params=query_params
        ) as response:
            if not response.ok:
                _raise_for_status(response.status, await response.text())
            return await response.json()

    SSE_DATA_PREFIX = "data: "

    async def _post_request_with_streaming(
        self,
        endpoint: str,
        request: AnyRequest,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        json_body = self._build_json_body(request, model)
        json_body["stream"] = True

        query_params = self._build_query_parameters()

        async with self.session.post(
            self.host + endpoint, json=json_body, params=query_params
        ) as response:
            if not response.ok:
                _raise_for_status(response.status, await response.text())

            async for stream_item in response.content:
                stream_item_as_str = stream_item.decode().strip()

                if not stream_item_as_str:
                    continue

                if not stream_item_as_str.startswith(self.SSE_DATA_PREFIX):
                    raise ValueError(
                        f"Stream item did not start with `{self.SSE_DATA_PREFIX}`. Was `{stream_item_as_str}"
                    )

                payload = stream_item_as_str[len(self.SSE_DATA_PREFIX) :]
                if payload == "[DONE]":
                    continue

                yield json.loads(payload)

    def _build_query_parameters(self) -> Mapping[str, str]:
        return {
            # cannot use str() here because we want lowercase true/false in query string
            # Also do not want to send the nice flag with every request if it is false
            **({"nice": "true"} if self.nice else {}),
        }

    def _build_json_body(
        self, request: AnyRequest, model: Optional[str]
    ) -> Dict[str, Any]:
        json_body = dict(request.to_json())

        if model is not None:
            json_body["model"] = model
        if self.hosting is not None:
            json_body["hosting"] = self.hosting
        if self.tags is not None:
            json_body["tags"] = self.tags
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

    async def chat(
        self,
        request: ChatRequest,
        model: str,
    ) -> ChatResponse:
        """Chat with a model.

        Parameters:
            request (ChatRequest, required):
                Parameters for the requested chat.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # create a chat request
            >>> request = ChatRequest(
                    messages=[Message(role="user", content="Hello, how are you?")],
                    model=model,
                )
            >>>
            >>> # chat with the model
            >>> result = await client.chat(request, model=model_name)
        """
        response = await self._post_request(
            "chat/completions",
            request,
            model,
        )
        return ChatResponse.from_json(response)

    async def complete_with_streaming(
        self,
        request: CompletionRequest,
        model: str,
    ) -> AsyncGenerator[CompletionResponseStreamItem, None]:
        """Generates streamed completions given a prompt.

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
            >>> result = await client.complete_with_streaming(request, model=model_name)
            >>>
            >>> # consume the completion stream
            >>> async for stream_item in result:
            >>>     do_something_with(stream_item)
        """
        async for stream_item_json in self._post_request_with_streaming(
            "complete",
            request,
            model,
        ):
            yield stream_item_from_json(stream_item_json)

    async def chat_with_streaming(
        self,
        request: ChatRequest,
        model: str,
    ) -> AsyncGenerator[Union[ChatStreamChunk, Usage], None]:
        """Generates streamed chat completions.

        The first yielded chunk contains the role, while subsequent chunks only contain the content delta.

        Parameters:
            request (ChatRequest, required):
                Parameters for the requested chat.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> # create a chat request
            >>> request = ChatRequest(
                    messages=[Message(role="user", content="Hello, how are you?")],
                    model=model,
                )
            >>>
            >>> # chat with the model
            >>> result = await client.chat_with_streaming(request, model=model_name)
            >>>
            >>> # consume the chat stream
            >>> async for stream_item in result:
            >>>     do_something_with(stream_item)
        """
        async for stream_item_json in self._post_request_with_streaming(
            "chat/completions",
            request,
            model,
        ):
            chunk = stream_chat_item_from_json(stream_item_json)
            if chunk is not None:
                yield chunk

    async def tokenize(
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

            model (string, required):
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

            model (string, required):
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
                Parameters for the requested semantic embedding.

            model (string, required):
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

    async def batch_semantic_embed(
        self,
        request: BatchSemanticEmbeddingRequest,
        model: Optional[str] = None,
        num_concurrent_requests: int = 1,
        batch_size: int = 100,
        progress_bar: bool = False,
    ) -> BatchSemanticEmbeddingResponse:
        """Embeds a sequence of texts or images and returns vectors in the same order as they
        were provided. If more than `batch_size` prompts are provided then this method will chunk them
        into batches of up to `batch_size` prompts that will be sent to the API.

        Parameters:
            request (BatchSemanticEmbeddingRequest, required):
                Parameters for the requested semantic embeddings.

            model (string, optional, default None):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

            num_concurrent_requests (int, optional, default 1):
                Maximum number of concurrent requests to send to the API.

            batch_size (int, optional, default 100):
                Number of prompts per batch sent to the API. This value must be between 1 and 100 (inclusive).

            progress_bar (bool, optional, default False):
                Whether to show a progress bar using tqdm.

        Examples:
            >>> # function for symmetric embedding
            >>> def embed_symmetric(texts: Sequence[str]):
                    # Create an embeddingrequest with the type set to symmetric
                    request = BatchSemanticEmbeddingRequest(
                        prompts=[Prompt.from_text(text) for text in texts],
                        representation=SemanticRepresentation.Symmetric
                    )
                    # create the embedding
                    result = client.batch_semantic_embed(request, model=model_name)
                    return result.embedding
        """
        if batch_size < 1 or batch_size > 100:
            raise ValueError(
                "`batch_semantic_embed` must be called with a `batch_size` between 1 and 100 (inclusive)"
            )

        responses: List[EmbeddingVector] = []
        model_version = ""

        # The API currently only supports batch semantic embedding requests with up to 100
        # prompts per batch. As a convenience for users, this function chunks larger requests.
        results = await self._gather_with_concurrency(
            "batch_semantic_embed",
            model,
            num_concurrent_requests,
            _generate_semantic_embedding_batches(request, batch_size),
            progress_bar,
        )
        num_tokens_prompt_total = 0
        for result in results:
            resp = BatchSemanticEmbeddingResponse.from_json(result)
            model_version = resp.model_version
            responses.extend(resp.embeddings)
            num_tokens_prompt_total += resp.num_tokens_prompt_total

        return BatchSemanticEmbeddingResponse(
            model_version=model_version,
            embeddings=responses,
            num_tokens_prompt_total=num_tokens_prompt_total,
        )

    async def evaluate(
        self,
        request: EvaluationRequest,
        model: str,
    ) -> EvaluationResponse:
        """Evaluates the model's likelihood to produce a completion given a prompt.

        Parameters:
            request (EvaluationRequest, required):
                Parameters for the requested evaluation.

            model (string, required):
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

    async def qa(self, request: QaRequest) -> QaResponse:
        """DEPRECATED: `qa` is deprecated and will be removed in the next major release. New
        methods of processing Q&A tasks will be provided before this is removed.

        Answers a question about documents.

        Parameters:
            request (QaRequest, required):
                Parameters for the qa request.

        Examples:
            >>> request = QaRequest(
                    query="Who likes pizza?",
                    documents=[Document.from_text("Andreas likes pizza.")],
                )
            >>> response = await client.qa(request, model="luminous-extended")
        """
        warnings.warn(
            "qa is deprecated and will be removed in the next major release. New methods of processing Q&A tasks will be provided before this is removed.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        response = await self._post_request("qa", request)
        return QaResponse.from_json(response)

    async def summarize(
        self,
        request: SummarizationRequest,
    ) -> SummarizationResponse:
        """DEPRECATED: `summarize` is deprecated and will be removed in the next major release. New
        methods of processing Summarization tasks will be provided before this is removed.

        Summarizes a document.

        Parameters:
            request (SummarizationRequest, required):
                Parameters for the requested summarization.
        Examples:
            >>> request = SummarizationRequest(
                    document=Document.from_text("Andreas likes pizza."),
                )
            >>> response = await client.summarize(request, model="luminous-extended")
        """
        warnings.warn(
            "summarize is deprecated and will be removed in the next major release. New methods of processing Q&A tasks will be provided before this is removed.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        response = await self._post_request(
            "summarize",
            request,
        )
        return SummarizationResponse.from_json(response)

    async def explain(
        self,
        request: ExplanationRequest,
        model: str,
    ) -> ExplanationResponse:
        """Better understand the source of a completion, specifically on how much each section of a
        prompt impacts each token of the completion.

        Parameters:
            request (ExplanationRequest, required):
                Parameters for the requested explanation.

            model (string, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others).
                Always the latest version of model is used.

        Examples:
            >>> request = ExplanationRequest(
                    prompt=Prompt.from_text("Andreas likes"),
                    target=" pizza."
                )
            >>> response = await client.explain(request, model="luminous-extended")
        """
        response = await self._post_request(
            "explain",
            request,
            model,
        )
        return ExplanationResponse.from_json(response)

    async def tokenizer(self, model: str) -> Tokenizer:
        """Returns a Tokenizer instance with the settings that were used to train the model.

        Examples:
            >>> tokenizer = await client.tokenizer(model="luminous-extended")
            >>> tokenized_prompt = tokenizer.encode("Hello world")
        """
        response = await self._get_request_text(f"models/{model}/tokenizer")
        return Tokenizer.from_str(response)

    # Based on: https://docs.aleph-alpha.com/changelog/2022/11/14/async-python-client/
    async def _gather_with_concurrency(
        self,
        endpoint: str,
        model: Optional[str],
        n: int,
        requests: Sequence[BatchSemanticEmbeddingRequest],
        progress_bar: bool,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(n)

        async def sem_task(request: BatchSemanticEmbeddingRequest):
            async with semaphore:
                return await self._post_request(endpoint, request, model)

        # asyncio.gather preserves order of awaitables in result list
        if progress_bar:
            return await tqdm.gather(*(sem_task(request) for request in requests))
        else:
            return await asyncio.gather(*(sem_task(request) for request in requests))


def _generate_semantic_embedding_batches(
    request: BatchSemanticEmbeddingRequest, batch_size: int = 100
) -> List[BatchSemanticEmbeddingRequest]:
    requests = []
    for batch_index in range(0, len(request.prompts), batch_size):
        batch = request.prompts[batch_index : batch_index + batch_size]
        requests.append(
            BatchSemanticEmbeddingRequest(
                prompts=batch,
                representation=request.representation,
                compress_to_size=request.compress_to_size,
                normalize=request.normalize,
                contextual_control_threshold=request.contextual_control_threshold,
                control_log_additive=request.control_log_additive,
            )
        )
    return requests

from collections import ChainMap
from typing import Any, Mapping, Optional, Union
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.completion import CompletionRequest, CompletionResponse
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
from aleph_alpha_client.evaluation import EvaluationRequest, EvaluationResponse
from aleph_alpha_client.explanation import ExplanationRequest
from aleph_alpha_client.qa import QaRequest, QaResponse
from aleph_alpha_client.tokenization import TokenizationRequest, TokenizationResponse
from aleph_alpha_client.summarization import SummarizationRequest, SummarizationResponse


class AlephAlphaModel:
    def __init__(
        self, client: AlephAlphaClient, model_name: str, hosting: Optional[str] = None
    ) -> None:
        """
        Construct a context object for a specific model.

        Parameters:
            client (AlephAlphaClient, required):
                An AlephAlphaClient object that holds the API host information and user credentials.

            model_name (str, required):
                Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.

            hosting (str, optional, default None):
                Determines in which datacenters the request may be processed.
                You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).

                Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
                own datacenters and on servers hosted with other providers. Choose this option for maximal availability.

                Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
                Choose this option for maximal data privacy.
        """

        self.client = client
        self.model_name = model_name
        self.hosting = hosting

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response_json = self.client.complete(
            model=self.model_name, hosting=self.hosting, **self.as_request_dict(request)
        )
        return CompletionResponse.from_json(response_json)

    def tokenize(self, request: TokenizationRequest) -> TokenizationResponse:
        response_json = self.client.tokenize(model=self.model_name, **request._asdict())
        return TokenizationResponse.from_json(response_json)

    def detokenize(self, request: DetokenizationRequest) -> DetokenizationResponse:
        response_json = self.client.detokenize(
            model=self.model_name, **request._asdict()
        )
        return DetokenizationResponse.from_json(response_json)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response_json = self.client.embed(
            model=self.model_name, hosting=self.hosting, **self.as_request_dict(request)
        )
        return EmbeddingResponse.from_json(response_json)

    def semantic_embed(
        self, request: SemanticEmbeddingRequest
    ) -> SemanticEmbeddingResponse:
        response_json = self.client.semantic_embed(
            model=self.model_name, hosting=self.hosting, request=request
        )
        return SemanticEmbeddingResponse.from_json(response_json)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        response_json = self.client.evaluate(
            model=self.model_name, hosting=self.hosting, **self.as_request_dict(request)
        )
        return EvaluationResponse.from_json(response_json)

    def qa(self, request: QaRequest) -> QaResponse:
        response_json = self.client.qa(
            model=self.model_name, hosting=self.hosting, **request._asdict()
        )
        return QaResponse.from_json(response_json)

    def _explain(self, request: ExplanationRequest) -> Mapping[str, Any]:
        return self.client._explain(
            model=self.model_name, hosting=self.hosting, request=request
        )

    def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        response_json = self.client.summarize(
            self.model_name, request, hosting=self.hosting
        )
        return SummarizationResponse.from_json(response_json)

    @staticmethod
    def as_request_dict(
        request: Union[CompletionRequest, EmbeddingRequest, EvaluationRequest]
    ) -> Mapping[str, Any]:
        return ChainMap({"prompt": request.prompt.items}, request._asdict())

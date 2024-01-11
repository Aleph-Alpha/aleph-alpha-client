from dataclasses import asdict, dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from aleph_alpha_client.prompt import Prompt


@dataclass(frozen=True)
class EmbeddingRequest:
    """
    Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

    Parameters:
        prompt
            The text and/or image(s) to be embedded.

        layers
            A list of layer indices from which to return embeddings.

            * Index 0 corresponds to the word embeddings used as input to the first transformer layer
            * Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
            * Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last layer etc.

        pooling
            Pooling operation to use.
            Pooling operations include:

            * mean: aggregate token embeddings across the sequence dimension using an average
            * max: aggregate token embeddings across the sequence dimension using a maximum
            * last_token: just use the last token
            * abs_max: aggregate token embeddings across the sequence dimension using a maximum of absolute values

        type
            Type of the embedding (e.g. symmetric or asymmetric)

        tokens
            Flag indicating whether the tokenized prompt is to be returned (True) or not (False)

        normalize
            Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.

            Note that at the moment this parameter does not yet have any effect. This will change as soon as the
            corresponding feature is available in the backend

        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.

        control_log_additive (bool, default True)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor

    Examples:
        >>> prompt = Prompt.from_text("This is an example.")
        >>> EmbeddingRequest(prompt=prompt, layers=[-1], pooling=["mean"])
    """

    prompt: Prompt
    layers: List[int]
    pooling: List[str]
    type: Optional[str] = None
    tokens: bool = False
    normalize: bool = False
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = True

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "prompt": self.prompt.to_json(),
        }

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingResponse:
    model_version: str
    num_tokens_prompt_total: int
    embeddings: Optional[Dict[Tuple[str, str], List[float]]]
    tokens: Optional[List[str]]
    message: Optional[str] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "EmbeddingResponse":
        return EmbeddingResponse(
            model_version=json["model_version"],
            embeddings={
                (layer, pooling): embedding
                for layer, pooling_dict in json["embeddings"].items()
                for pooling, embedding in pooling_dict.items()
            },
            tokens=json.get("tokens"),
            message=json.get("message"),
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
        )


class SemanticRepresentation(Enum):
    """
    Available types of semantic representations that prompts can be embedded with.

    Symmetric:
        `Symmetric` is useful for comparing prompts to each other, in use cases such as clustering, classification, similarity, etc. `Symmetric` embeddings should be compared with other `Symmetric` embeddings.
    Document:
        `Document` and `Query` are used together in use cases such as search where you want to compare shorter queries against larger documents.

        `Document` embeddings are optimized for larger pieces of text to compare queries against.
    Query:
        `Document` and `Query` are used together in use cases such as search where you want to compare shorter queries against larger documents.

        `Query` embeddings are optimized for shorter texts, such as questions or keywords.
    """

    Symmetric = "symmetric"
    Document = "document"
    Query = "query"


@dataclass(frozen=True)
class SemanticEmbeddingRequest:
    """
    Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

    Parameters:
        prompt
            The text and/or image(s) to be embedded.
        representation
            Semantic representation to embed the prompt with.
        compress_to_size
            Options available: 128

            The default behavior is to return the full embedding, but you can optionally request an embedding compressed to a smaller set of dimensions.

            Full embedding sizes for supported models:
              - luminous-base: 5120

            The 128 size is expected to have a small drop in accuracy performance (4-6%), with the benefit of being much smaller, which makes comparing these embeddings much faster for use cases where speed is critical.

            The 128 size can also perform better if you are embedding really short texts or documents.

        normalize
            Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.

            Note that at the moment this parameter does not yet have any effect. This will change as soon as the
            corresponding feature is available in the backend

        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.

        control_log_additive (bool, default True)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor

    Examples
        >>> texts = [
                "deep learning",
                "artificial intelligence",
                "deep diving",
                "artificial snow",
            ]
        >>> # Texts to compare
        >>> embeddings = []
        >>> for text in texts:
                request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), representation=SemanticRepresentation.Symmetric)
                result = model.semantic_embed(request)
                embeddings.append(result.embedding)
    """

    prompt: Prompt
    representation: SemanticRepresentation
    compress_to_size: Optional[int] = None
    normalize: bool = False
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = True

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "representation": self.representation.value,
            "prompt": self.prompt.to_json(),
        }

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchSemanticEmbeddingRequest:
    """
    Embeds multiple multi-modal prompts and returns their embeddings in the same order as they were supplied.

    Parameters:
        prompts
            A list of texts and/or images to be embedded.
        representation
            Semantic representation to embed the prompt with.
        compress_to_size
            Options available: 128

            The default behavior is to return the full embedding, but you can optionally request an embedding compressed to a smaller set of dimensions.

            Full embedding sizes for supported models:
              - luminous-base: 5120

            The 128 size is expected to have a small drop in accuracy performance (4-6%), with the benefit of being much smaller, which makes comparing these embeddings much faster for use cases where speed is critical.

            The 128 size can also perform better if you are embedding really short texts or documents.

        normalize
            Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.

            Note that at the moment this parameter does not yet have any effect. This will change as soon as the
            corresponding feature is available in the backend

        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.

        control_log_additive (bool, default True)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor

    Examples
        >>> texts = [
                "deep learning",
                "artificial intelligence",
                "deep diving",
                "artificial snow",
            ]
        >>> # Texts to compare
        >>> request = BatchSemanticEmbeddingRequest(prompts=[Prompt.from_text(text) for text in texts], representation=SemanticRepresentation.Symmetric)
            result = model.batch_semantic_embed(request)
    """

    prompts: Sequence[Prompt]
    representation: SemanticRepresentation
    compress_to_size: Optional[int] = None
    normalize: bool = False
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = True

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "representation": self.representation.value,
            "prompts": [prompt.to_json() for prompt in self.prompts],
        }

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


EmbeddingVector = List[float]


@dataclass(frozen=True)
class SemanticEmbeddingResponse:
    """
    Response of a semantic embedding request

    Parameters:
        model_version
            Model name and version (if any) of the used model for inference
        embedding
            A list of floats that can be used to compare against other embeddings.
        message
            This field is no longer used.
    """

    model_version: str
    embedding: EmbeddingVector
    num_tokens_prompt_total: int
    message: Optional[str] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "SemanticEmbeddingResponse":
        return SemanticEmbeddingResponse(
            model_version=json["model_version"],
            embedding=json["embedding"],
            message=json.get("message"),
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
        )


@dataclass(frozen=True)
class BatchSemanticEmbeddingResponse:
    """
    Response of a batch semantic embedding request

    Parameters:
        model_version
            Model name and version (if any) of the used model for inference
        embeddings
            A list of embeddings.
    """

    model_version: str
    embeddings: Sequence[EmbeddingVector]
    num_tokens_prompt_total: int

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "BatchSemanticEmbeddingResponse":
        return BatchSemanticEmbeddingResponse(
            model_version=json["model_version"],
            embeddings=json["embeddings"],
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
        )

    def to_json(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "embeddings": [embedding for embedding in self.embeddings],
        }

    @staticmethod
    def _from_model_version_and_embeddings(
        model_version: str,
        embeddings: Sequence[EmbeddingVector],
        num_tokens_prompt_total: int,
    ) -> "BatchSemanticEmbeddingResponse":
        return BatchSemanticEmbeddingResponse(
            model_version=model_version,
            embeddings=embeddings,
            num_tokens_prompt_total=num_tokens_prompt_total,
        )

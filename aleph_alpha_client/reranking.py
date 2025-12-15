from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RerankRequest:
    """Request for reranking documents against a query.

    This endpoint takes in a query and a list of documents and produces an array
    with each document assigned a relevance score.

    Parameters:
        query (str, required):
            The query to rerank the documents against.

        documents (Sequence[str], required):
            The list of documents to rerank.

        top_n (int, optional):
            The number of documents to return. Defaults to the number of documents
            if not provided.

    Examples:
        >>> request = RerankRequest(
        ...     query="What is the capital of France?",
        ...     documents=[
        ...         "The capital of Brazil is Brasilia.",
        ...         "The capital of France is Paris.",
        ...         "Horses and cows are both animals.",
        ...     ],
        ...     top_n=2,
        ... )
        >>> response = client.rerank(request, model="your-reranker-model")
    """

    query: str
    documents: Sequence[str]
    top_n: Optional[int] = None

    def to_json(self) -> Mapping[str, Any]:
        """Convert the request to a JSON-serializable dictionary."""
        json_request: Dict[str, Any] = {
            "query": self.query,
            "documents": list(self.documents),
        }
        if self.top_n is not None:
            json_request["top_n"] = self.top_n
        return json_request


@dataclass(frozen=True)
class RerankResult:
    """A single reranked document result.

    Parameters:
        index (int):
            The index of the document in the original list of documents.

        relevance_score (float):
            The relevance score of the document.
    """

    index: int
    relevance_score: float

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "RerankResult":
        return RerankResult(
            index=json["index"],
            relevance_score=json["relevance_score"],
        )


@dataclass(frozen=True)
class RerankUsage:
    """Usage statistics for the rerank request.

    Parameters:
        completion_tokens (int):
            Number of tokens in the generated completion. Will always be 0 for rerank tasks.

        prompt_tokens (int):
            Number of tokens in the prompt. Will always be 0 for rerank tasks.

        total_tokens (int):
            Total number of tokens used in the request.
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "RerankUsage":
        return RerankUsage(
            completion_tokens=json["completion_tokens"],
            prompt_tokens=json["prompt_tokens"],
            total_tokens=json["total_tokens"],
        )


@dataclass(frozen=True)
class RerankResponse:
    """Response from a rerank request.

    Parameters:
        results (List[RerankResult]):
            The reranked results, each containing the original document index
            and its relevance score.

        usage (RerankUsage):
            Usage statistics for the request.
    """

    results: List[RerankResult]
    usage: RerankUsage

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "RerankResponse":
        return RerankResponse(
            results=[RerankResult.from_json(r) for r in json["results"]],
            usage=RerankUsage.from_json(json["usage"]),
        )

from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence

from aleph_alpha_client.prompt import Prompt


class SearchRequest(NamedTuple):
    """
    Describes a search request

    Parameters:
        query (Prompt, required):
            The search query. Content items in corpus will be ordered by similarity to this query.

        corpus (Mapping[str, Prompt], required):
            The corpus of content to be searched.

        max_results (int, optional):
            Limits the amount of returned search results if not None.
            If None, all results that match the `min_score` criterion are returned.

        min_score (float, optional):
            Limits the minimal score of returned results if not None.
            If None, all results that match the `max_results` critreion are returned.

    Examples:
        >>> request = SearchRequest(
                query=Prompt.from_text("banana"),
                corpus={"id0": Prompt.from_text("apple"), "id1": Prompt.from_text("banana")},
            )
    """

    query: Prompt
    corpus: Mapping[str, Prompt]
    max_results: Optional[int] = None
    min_score: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_json(),
            "corpus": {k: v.to_json() for k, v in self.corpus.items()},
            "max_results": self.max_results,
            "min_score": self.min_score,
        }


class SearchResult(NamedTuple):
    """
    Describes the search results

    Parameters:
        id (str, required):
            The id of the content as given in the corpus of the SearchRequest.

        score (float, required):
            The semantic similarity score of the content with the given id to the query.
    """

    id: str
    score: float


class SearchResponse(NamedTuple):
    model_version: str
    results: Sequence[SearchResult]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "SearchResponse":
        return SearchResponse(
            model_version=json["model_version"],
            results=[SearchResult(**item) for item in json["results"]],
        )

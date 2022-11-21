from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from aleph_alpha_client.prompt import Prompt, _to_serializable_prompt


class SearchRequest(NamedTuple):
    """
    Describes a search request

    Parameters:
        query (Prompt, required):
            The search query. Content items in corpus will be ordered by similarity to this query.

        corpus (Mapping[str, Prompt], required):
            The corpus of content to be searched.

    Examples
        >>> request = SearchRequest(
                query=Prompt.from_text("banana"),
                corpus={"id0": Prompt.from_text("apple"), "id1": Prompt.from_text("banana")},
            )
    """

    query: Prompt
    corpus: Mapping[str, Prompt]

    def to_json(self) -> Dict[str, Any]:
        return {
            "query": _to_serializable_prompt(self.query.items),
            "corpus": {
                k: _to_serializable_prompt(v.items) for k, v in self.corpus.items()
            },
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

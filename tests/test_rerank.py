import pytest

from aleph_alpha_client.reranking import RerankRequest, RerankResponse
from tests.conftest import GenericClient


@pytest.mark.vcr
@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_can_rerank_documents(
    generic_client: GenericClient, rerank_model_name: str
):
    request = RerankRequest(
        query="What is the capital of France?",
        documents=[
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
            "Horses and cows are both animals.",
        ],
    )

    response = await generic_client.rerank(request, model=rerank_model_name)
    assert isinstance(response, RerankResponse)
    assert len(response.results) == 3
    for result in response.results:
        assert result.index >= 0
        assert result.index < 3
        assert isinstance(result.relevance_score, float)

    # The document about Paris should have the highest relevance score
    highest_result = response.results[0]
    assert highest_result.relevance_score > 0
    assert highest_result.index == 1

    assert response.usage.total_tokens > 0


@pytest.mark.vcr
@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_can_rerank_with_top_n(
    generic_client: GenericClient, rerank_model_name: str
):
    request = RerankRequest(
        query="What is the capital of France?",
        documents=[
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
            "Horses and cows are both animals.",
        ],
        top_n=2,
    )

    response = await generic_client.rerank(request, model=rerank_model_name)
    assert isinstance(response, RerankResponse)
    assert len(response.results) == 2


@pytest.mark.vcr
@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_rerank_returns_results_ordered_by_relevance(
    generic_client: GenericClient, rerank_model_name: str
):
    request = RerankRequest(
        query="What is the capital of France?",
        documents=[
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
            "Horses and cows are both animals.",
        ],
    )

    response = await generic_client.rerank(request, model=rerank_model_name)

    # Verify results are sorted by relevance score in descending order
    scores = [r.relevance_score for r in response.results]
    assert scores == sorted(scores, reverse=True)

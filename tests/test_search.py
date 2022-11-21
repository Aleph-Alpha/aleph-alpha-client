import pytest
from aleph_alpha_client import AsyncClient, Prompt, SearchRequest
from .common import async_client


@pytest.mark.needs_api
async def test_complete_with_token_ids(
    async_client: AsyncClient,
):
    request = SearchRequest(
        query=Prompt.from_text("banana"),
        corpus={"jump": Prompt.from_text("jump"), "banana": Prompt.from_text("banana")},
    )

    response = await async_client._search(request)
    assert len(response.results) == 2
    assert dict(response.results)["banana"] > dict(response.results)["jump"]

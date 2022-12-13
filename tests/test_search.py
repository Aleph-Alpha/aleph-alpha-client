import pytest
from aleph_alpha_client import AsyncClient, Prompt, SearchRequest, Client
from .common import async_client, sync_client


async def test_async_basic_functionality(
    async_client: AsyncClient,
):
    request = SearchRequest(
        query=Prompt.from_text("banana"),
        corpus={"jump": Prompt.from_text("jump"), "banana": Prompt.from_text("banana")},
    )

    response = await async_client._search(request)
    assert len(response.results) == 2
    assert dict(response.results)["banana"] > dict(response.results)["jump"]


async def test_max_results(
    async_client: AsyncClient,
):
    request = SearchRequest(
        query=Prompt.from_text("banana"),
        corpus={"jump": Prompt.from_text("jump"), "banana": Prompt.from_text("banana")},
        max_results=1,
    )

    response = await async_client._search(request)
    assert len(response.results) == 1
    assert "banana" in dict(response.results)


async def test_min_score(
    async_client: AsyncClient,
):
    request = SearchRequest(
        query=Prompt.from_text("banana"),
        corpus={"jump": Prompt.from_text("jump"), "banana": Prompt.from_text("banana")},
        min_score=0.5,
    )

    response = await async_client._search(request)
    assert len(response.results) == 1
    assert "banana" in dict(response.results)


def test_sync_basic_functionality(
    sync_client: Client,
):
    request = SearchRequest(
        query=Prompt.from_text("banana"),
        corpus={"jump": Prompt.from_text("jump"), "banana": Prompt.from_text("banana")},
    )

    response = sync_client._search(request)
    assert len(response.results) == 2
    assert dict(response.results)["banana"] > dict(response.results)["jump"]

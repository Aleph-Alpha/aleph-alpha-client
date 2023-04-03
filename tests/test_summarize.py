import pytest
from aleph_alpha_client import Document, SummarizationRequest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client

from tests.common import (
    sync_client,
    async_client,
    model_name,
)

# AsyncClient


@pytest.mark.system_test
async def test_can_summarize_with_async_client(async_client: AsyncClient):
    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = await async_client.summarize(request)
    assert response.summary is not None


# Client


@pytest.mark.system_test
def test_summarize(sync_client: Client):
    # when posting a Summarization request
    request = SummarizationRequest(
        document=Document.from_prompt(["Andreas likes pizza."]),
    )

    response = sync_client.summarize(request)

    # the response should exist and be in the form of a named tuple class
    assert response.summary is not None


@pytest.mark.system_test
def test_text(sync_client: Client):
    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = sync_client.summarize(request)

    assert response.summary is not None

import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.document import Document
from aleph_alpha_client.qa import QaRequest

from tests.common import (
    sync_client,
    model_name,
    async_client,
)

# AsyncClient


async def test_can_qa_with_async_client(async_client: AsyncClient):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    response = await async_client.qa(request)
    assert len(response.answers) == 1
    assert response.answers[0].score > 0.0


# Client


def test_qa(sync_client: Client):
    # when posting a QA request with a QaRequest object
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
    )

    response = sync_client.qa(request)

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 1


def test_qa_no_answer_found(sync_client: Client):
    # when posting a QA request with a QaRequest object
    request = QaRequest(
        query="Who likes pizza?",
        documents=[],
    )

    response = sync_client.qa(request)

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 0


def test_text(sync_client: Client):
    # when posting an illegal request
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    # then we expect an exception tue to a bad request response from the API
    response = sync_client.qa(request)

    # The response should exist in the form of a json dict
    assert len(response.answers) == 1
    assert response.answers[0].score > 0.5

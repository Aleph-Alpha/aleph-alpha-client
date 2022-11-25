import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.document import Document
from aleph_alpha_client.qa import QaRequest

from tests.common import (
    sync_client,
    client,
    model_name,
    luminous_extended,
    qa_checkpoint_name,
    async_client,
)

# AsyncClient


@pytest.mark.needs_api
async def test_can_qa_with_async_client(async_client: AsyncClient):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    response = await async_client.qa(request, model="luminous-extended")
    assert len(response.answers) == 1
    assert response.model_version is not None
    assert response.answers[0].score > 0.0


@pytest.mark.needs_api
async def test_can_qa_with_async_client_against_checkpoint(
    async_client: AsyncClient, qa_checkpoint_name: str
):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    response = await async_client.qa(request, checkpoint=qa_checkpoint_name)
    assert len(response.answers) == 1
    assert response.model_version is not None
    assert response.answers[0].score > 0.5


# Client


@pytest.mark.needs_api
def test_qa(sync_client: Client):
    # when posting a QA request with a QaRequest object
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
    )

    response = sync_client.qa(request, model="luminous-extended")

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 1
    assert response.model_version is not None


@pytest.mark.needs_api
def test_qa_against_checkpoint(sync_client: Client, qa_checkpoint_name: str):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
    )

    response = sync_client.qa(request, checkpoint=qa_checkpoint_name)

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 1
    assert response.model_version is not None


@pytest.mark.needs_api
def test_qa_no_answer_found(sync_client: Client):
    # when posting a QA request with a QaRequest object
    request = QaRequest(
        query="Who likes pizza?",
        documents=[],
    )

    response = sync_client.qa(request, model="luminous-extended")

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 0
    assert response.model_version is not None


@pytest.mark.needs_api
def test_text(sync_client: Client):
    # when posting an illegal request
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    # then we expect an exception tue to a bad request response from the API
    response = sync_client.qa(request, model="luminous-extended")

    # The response should exist in the form of a json dict
    assert len(response.answers) == 1
    assert response.model_version is not None
    assert response.answers[0].score > 0.5


# AlephAlphaClient


@pytest.mark.needs_api
def test_qa_with_client(client: AlephAlphaClient):
    model_name = "luminous-extended"
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting a QA request with explicit parameters
    response = client.qa(
        model_name,
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
    )

    # The response should exist in the form of a json dict
    assert len(response["answers"]) == 1
    assert response["model_version"] is not None


@pytest.mark.needs_api
def test_qa_with_client_against_checkpoint(
    client: AlephAlphaClient, qa_checkpoint_name: str
):
    # given a client
    assert qa_checkpoint_name in map(
        lambda checkpoint: checkpoint["name"], client.available_checkpoints()
    )

    # when posting a QA request with explicit parameters
    response = client.qa(
        model=None,
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
        checkpoint=qa_checkpoint_name,
    )

    # The response should exist in the form of a json dict
    assert len(response["answers"]) == 1
    assert response["model_version"] is not None

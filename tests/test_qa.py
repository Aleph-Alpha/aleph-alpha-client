import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.document import Document
from aleph_alpha_client.qa import QaRequest

from tests.common import (
    sync_client,
    client,
    model_name,
    luminous_extended,
    async_client,
)

# AsyncClient


@pytest.mark.system_test
async def test_can_qa_with_async_client(async_client: AsyncClient):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    response = await async_client.qa(request, model="luminous-extended")
    assert len(response.answers) == 1
    assert response.model_version is not None
    assert response.answers[0].score > 0.0


# Client


@pytest.mark.system_test
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


@pytest.mark.system_test
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


@pytest.mark.system_test
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


@pytest.mark.system_test
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


def test_can_send_beta_request_and_no_model(sync_client: Client):
    # when posting a QA request with a QaRequest object
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_prompt(["Andreas likes pizza."])],
    )

    response = sync_client.qa(request, beta=True)

    # the response should exist and be in the form of a named tuple class
    assert len(response.answers) == 1
    assert response.model_version is not None


async def test_can_send_async_beta_request_and_no_model(async_client: AsyncClient):
    request = QaRequest(
        query="Who likes pizza?",
        documents=[Document.from_text("Andreas likes pizza.")],
    )

    response = await async_client.qa(request, beta=True)
    assert len(response.answers) == 1
    assert response.model_version is not None
    assert response.answers[0].score > 0.0

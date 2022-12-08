from aleph_alpha_client import (
    AlephAlphaClient,
    AlephAlphaModel,
    Document,
    SummarizationRequest,
)
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client

from tests.common import (
    sync_client,
    async_client,
    client,
    model_name,
    luminous_extended,
    summarization_checkpoint_name,
)
import pytest

# AsyncClient


@pytest.mark.system_test
async def test_can_summarize_with_async_client(async_client: AsyncClient):
    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = await async_client.summarize(request, model="luminous-extended")
    assert response.summary is not None
    assert response.model_version is not None


@pytest.mark.system_test
async def test_can_summarize_with_async_client_against_checkpoint(
    async_client: AsyncClient,
    summarization_checkpoint_name: str,
):
    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = await async_client.summarize(
        request, checkpoint=summarization_checkpoint_name
    )
    assert response.summary is not None
    assert response.model_version is not None


# Client


@pytest.mark.system_test
def test_summarize(sync_client: Client):
    # when posting a Summarization request
    request = SummarizationRequest(
        document=Document.from_prompt(["Andreas likes pizza."]),
    )

    response = sync_client.summarize(request, model="luminous-extended")

    # the response should exist and be in the form of a named tuple class
    assert response.summary is not None
    assert response.model_version is not None


@pytest.mark.system_test
def test_summarize_against_checkpoint(
    sync_client: Client, summarization_checkpoint_name: str
):
    request = SummarizationRequest(
        document=Document.from_prompt(["Andreas likes pizza."]),
    )
    response = sync_client.summarize(request, checkpoint=summarization_checkpoint_name)

    assert response.summary is not None
    assert response.model_version is not None


@pytest.mark.system_test
def test_text(sync_client: Client):
    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = sync_client.summarize(request, model="luminous-extended")

    assert response.summary is not None
    assert response.model_version is not None


# AlephAlphaClient


@pytest.mark.system_test
def test_summarization_with_client(client: AlephAlphaClient):
    model_name = "luminous-extended"
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting a Summarization request
    response = client.summarize(
        "luminous-extended",
        SummarizationRequest(
            document=Document.from_prompt(["Andreas likes pizza."]),
        ),
    )

    # The response should exist in the form of a json dict
    assert response["summary"] is not None
    assert response["model_version"] is not None


@pytest.mark.system_test
def test_summarization_with_client_against_checkpoint(
    client: AlephAlphaClient, summarization_checkpoint_name
):
    # when posting a Summarization request
    response = client.summarize(
        model=None,
        request=SummarizationRequest(
            document=Document.from_prompt(["Andreas likes pizza."]),
        ),
        checkpoint=summarization_checkpoint_name,
    )

    # The response should exist in the form of a json dict
    assert response["summary"] is not None
    assert response["model_version"] is not None

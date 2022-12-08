import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt

from tests.common import (
    client,
    sync_client,
    async_client,
    checkpoint_name,
    model_name,
    model,
    checkpoint_name,
    alt_complete_checkpoint_name,
    alt_complete_adapter_name,
)


# AsyncClient


@pytest.mark.system_test
async def test_can_complete_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    response = await async_client.complete(request, model=model_name)
    assert len(response.completions) == 1
    assert response.model_version is not None


@pytest.mark.system_test
async def test_can_complete_with_async_client_against_checkpoint(
    async_client: AsyncClient, checkpoint_name: str
):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    response = await async_client.complete(request, checkpoint=checkpoint_name)
    assert len(response.completions) == 1
    assert response.model_version is not None


async def test_can_complete_with_async_client_against_checkpoint_and_adapter(
    async_client: AsyncClient,
    alt_complete_checkpoint_name: str,
    alt_complete_adapter_name: str,
):
    request = CompletionRequest(
        prompt=Prompt.from_text("Hello, World!\n"),
        maximum_tokens=7,
    )

    response = await async_client.complete(
        request,
        checkpoint=alt_complete_checkpoint_name,
        adapter=alt_complete_adapter_name,
    )
    assert len(response.completions) == 1
    assert response.model_version is not None


# Client


@pytest.mark.system_test
def test_complete(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
        tokens=False,
        log_probs=0,
        logit_bias={1: 2.0},
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


@pytest.mark.system_test
def test_complete_with_token_ids(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_tokens([49222, 2998]),  # Hello world
        maximum_tokens=32,
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


@pytest.mark.system_test
def test_complete_against_checkpoint(sync_client: Client, checkpoint_name: str):

    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
        tokens=False,
        log_probs=0,
        logit_bias={1: 2.0},
    )

    response = sync_client.complete(request, checkpoint=checkpoint_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


async def test_can_complete_with_sync_client_against_checkpoint_and_adapter(
    sync_client: Client,
    alt_complete_checkpoint_name: str,
    alt_complete_adapter_name: str,
):
    request = CompletionRequest(
        prompt=Prompt.from_text("Hello, World!\n"),
        maximum_tokens=7,
    )

    response = sync_client.complete(
        request,
        checkpoint=alt_complete_checkpoint_name,
        adapter=alt_complete_adapter_name,
    )
    assert len(response.completions) == 1
    assert response.model_version is not None


# AlephAlphaClient


@pytest.mark.system_test
def test_complete_with_deprecated_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    response = client.complete(
        model=None,
        prompt=[""],
        maximum_tokens=7,
        tokens=False,
        log_probs=0,
        checkpoint=checkpoint_name,
    )

    assert len(response["completions"]) == 1
    assert response["model_version"] is not None


# AlephAlphaModel


@pytest.mark.system_test
def test_deprecated_complete(model: AlephAlphaModel):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
        tokens=False,
        log_probs=0,
        logit_bias={1: 2.0},
    )

    response = model.complete(request)

    assert len(response.completions) == 1
    assert response.model_version is not None

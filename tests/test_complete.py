import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt, Text, TextControl

from tests.common import (
    client,
    sync_client,
    async_client,
    model_name,
    model,
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


# Client


@pytest.mark.system_test
def test_complete(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt(
            [
                Text(
                    "Hello, World!",
                    controls=[TextControl(start=1, length=5, factor=0.5)],
                )
            ]
        ),
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

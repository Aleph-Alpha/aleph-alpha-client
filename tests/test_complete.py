import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt

from tests.common import (
    client,
    sync_client,
    checkpoint_name,
    model_name,
    model,
    checkpoint_name,
)


@pytest.mark.needs_api
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


@pytest.mark.needs_api
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


@pytest.mark.needs_api
def test_complete_with_token_ids(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_tokens([49222, 2998]),  # Hello world
        maximum_tokens=32,
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


@pytest.mark.needs_api
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


@pytest.mark.needs_api
def test_complete_with_client_against_checkpoint(
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

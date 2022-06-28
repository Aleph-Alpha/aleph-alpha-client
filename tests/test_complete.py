import json
from multiprocessing.sharedctypes import Value
import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt

from tests.common import client, model_name, model


def test_complete(model: AlephAlphaModel):
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


def test_complete_with_client(client: AlephAlphaClient, model_name: str):
    response = client.complete(
        model_name, prompt=[""], maximum_tokens=7, tokens=False, log_probs=0
    )

    assert len(response["completions"]) == 1
    assert response["model_version"] is not None


def test_complete_fails(model: AlephAlphaModel):
    # given a client
    assert model.model_name in (model["name"] for model in model.client.available_models())

    # when posting an illegal request
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=-1,
        tokens=False,
        log_probs=0,
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = model.complete(request)

    assert e.value.args[0] == 400

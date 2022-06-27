from multiprocessing.sharedctypes import Value
import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.completion import CompletionRequest

from tests.common import client, model


def test_complete(client: AlephAlphaClient, model: str):
    response = client.complete(
        model,
        hosting="cloud",
        request=CompletionRequest(
            prompt="",
            maximum_tokens=7,
            tokens=False,
            log_probs=0,
            logit_bias={1: 2.0},
        ),
    )

    assert len(response.completions) == 1
    assert response.model_version is not None


def test_complete_with_explicit_parameters(client: AlephAlphaClient, model: str):
    response = client.complete(
        model, prompt="", maximum_tokens=7, tokens=False, log_probs=0
    )

    assert len(response["completions"]) == 1
    assert response["model_version"] is not None


def test_complete_fails(client: AlephAlphaClient, model: str):
    # given a client
    assert model in (model["name"] for model in client.available_models())

    # when posting an illegal request
    request = CompletionRequest(
        prompt="",
        maximum_tokens=-1,
        tokens=False,
        log_probs=0,
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = client.complete(model, hosting="cloud", request=request)

    assert e.value.args[0] == 400

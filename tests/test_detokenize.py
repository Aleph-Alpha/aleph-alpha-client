import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import client, model


def test_detokenize(client: AlephAlphaClient, model: str):
    response = client.detokenize(model, request=DetokenizationRequest([4711]))

    assert response.result is not None


def test_detokenize_with_explicit_parameters(client: AlephAlphaClient, model: str):
    response = client.detokenize(model, token_ids=[4711, 42])

    assert response["result"] is not None


def test_detokenize_fails(client: AlephAlphaClient, model: str):
    # given a client
    assert model in map(lambda model: model["name"], client.available_models())

    # when posting an illegal request
    request = DetokenizationRequest([])

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(Exception):
        response = client.detokenize(
            model,
            request=request,
        )

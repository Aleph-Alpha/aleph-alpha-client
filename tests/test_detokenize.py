import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import client, model_name


def test_detokenize(client: AlephAlphaClient, model_name: str):
    response = client.detokenize(model_name, request=DetokenizationRequest([4711]))

    assert response.result is not None


def test_detokenize_with_explicit_parameters(client: AlephAlphaClient, model_name: str):
    response = client.detokenize(model_name, token_ids=[4711, 42])

    assert response["result"] is not None


def test_detokenize_fails(client: AlephAlphaClient, model_name: str):
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting an illegal request
    request = DetokenizationRequest([])

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = client.detokenize(
            model_name,
            request=request,
        )

    e.value.args[0] == 400
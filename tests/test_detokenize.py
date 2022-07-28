import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import client, model_name, model


def test_detokenize(model: AlephAlphaModel):
    response = model.detokenize(DetokenizationRequest([4711]))

    assert response.result is not None


def test_detokenize_with_client(client: AlephAlphaClient, model_name: str):
    response = client.detokenize(model_name, token_ids=[4711, 42])

    assert response["result"] is not None


def test_detokenize_fails(model: AlephAlphaModel):
    # given a client
    assert model.model_name in map(
        lambda model: model["name"], model.client.available_models()
    )

    # when posting an illegal request
    request = DetokenizationRequest([])

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = model.detokenize(request=request)

    assert e.value.args[0] == 400

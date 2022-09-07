import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import client, model_name, model, checkpoint_name


def test_tokenize(model: AlephAlphaModel):
    response = model.tokenize(
        request=TokenizationRequest("Hello", tokens=True, token_ids=True)
    )

    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


def test_tokenize_against_checkpoint(client: AlephAlphaClient, checkpoint_name: str):
    model = AlephAlphaModel(client, model_name=None, checkpoint_name=checkpoint_name)

    response = model.tokenize(
        request=TokenizationRequest("Hello", tokens=True, token_ids=True)
    )

    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


def test_tokenize_with_client_against_model(client: AlephAlphaClient, model_name: str):
    response = client.tokenize(model_name, prompt="Hello", tokens=True, token_ids=True)

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1


def test_tokenize_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    response = client.tokenize(
        model=None,
        prompt="Hello",
        tokens=True,
        token_ids=True,
        checkpoint=checkpoint_name,
    )

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1


def test_tokenize_fails(model: AlephAlphaModel):
    # given a client
    assert model.model_name in map(
        lambda model: model["name"], model.client.available_models()
    )

    # when posting an illegal request
    request = TokenizationRequest("hello", False, False)

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = model.tokenize(request)

    assert e.value.args[0] == 400

import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import client, model_name


def test_tokenize(client: AlephAlphaClient, model_name: str):
    response = client.tokenize(model_name, request=TokenizationRequest("Hello", True, True))

    assert len(response.tokens) == 1
    assert len(response.token_ids) == 1


def test_tokenize_with_explicit_parameters(client: AlephAlphaClient, model_name: str):
    response = client.tokenize(model_name, prompt="Hello", tokens=True, token_ids=True)

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1


def test_tokenize_fails(client: AlephAlphaClient, model_name: str):
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting an illegal request
    request = TokenizationRequest("hello", False, False)

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = client.tokenize(
            model_name,
            request=request,
        )

    assert e.value.args[0] == 400
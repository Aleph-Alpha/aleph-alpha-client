from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import client, model


def test_tokenize(client: AlephAlphaClient, model : str): 
    response = client.tokenize(model, request=TokenizationRequest("Hello", True, True))

    assert len(response.tokens) == 1
    assert len(response.token_ids) == 1


def test_tokenize_with_explicit_parameters(client: AlephAlphaClient, model : str): 
    response = client.tokenize(model, prompt="Hello", tokens=True, token_ids=True)

    assert len(response['tokens']) == 1
    assert len(response['token_ids']) == 1

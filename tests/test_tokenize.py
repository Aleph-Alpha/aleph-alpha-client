from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import client, model


def test_tokenize(client: AlephAlphaClient, model : str): 
    response = client.tokenize(model, TokenizationRequest("Hello", True, True))

    assert len(response.tokens) == 1
    assert len(response.token_ids) == 1

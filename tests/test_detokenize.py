import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import client, model_name, model, checkpoint_name, sync_client


@pytest.mark.needs_api
def test_detokenize(sync_client: Client, model_name: str):
    response = sync_client.detokenize(DetokenizationRequest([4711]), model=model_name)

    assert response.result is not None


@pytest.mark.needs_api
def test_detokenize_against_checkpoint(sync_client: Client, checkpoint_name: str):
    response = sync_client.detokenize(
        DetokenizationRequest([4711]), checkpoint=checkpoint_name
    )

    assert response.result is not None


@pytest.mark.needs_api
def test_detokenize_with_client(client: AlephAlphaClient, model_name: str):
    response = client.detokenize(model_name, token_ids=[4711, 42])

    assert response["result"] is not None


@pytest.mark.needs_api
def test_detokenize_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    response = client.detokenize(
        model=None, checkpoint=checkpoint_name, token_ids=[4711, 42]
    )

    assert response["result"] is not None

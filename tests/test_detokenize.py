import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import (
    client,
    model_name,
    model,
    checkpoint_name,
    sync_client,
    async_client,
)


# AsyncClient


@pytest.mark.system_test
async def test_can_detokenization_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = DetokenizationRequest(token_ids=[2, 3, 4])

    response = await async_client.detokenize(request, model=model_name)
    assert len(response.result) > 0


@pytest.mark.system_test
async def test_can_detokenization_with_async_client_with_checkpoint(
    async_client: AsyncClient, checkpoint_name: str
):
    request = DetokenizationRequest(token_ids=[2, 3, 4])

    response = await async_client.detokenize(request, checkpoint=checkpoint_name)
    assert len(response.result) > 0


# Client


@pytest.mark.system_test
def test_detokenize(sync_client: Client, model_name: str):
    response = sync_client.detokenize(DetokenizationRequest([4711]), model=model_name)

    assert response.result is not None


@pytest.mark.system_test
def test_detokenize_against_checkpoint(sync_client: Client, checkpoint_name: str):
    response = sync_client.detokenize(
        DetokenizationRequest([4711]), checkpoint=checkpoint_name
    )

    assert response.result is not None


# AlephAlphaClient


@pytest.mark.system_test
def test_detokenize_with_client(client: AlephAlphaClient, model_name: str):
    response = client.detokenize(model_name, token_ids=[4711, 42])

    assert response["result"] is not None


@pytest.mark.system_test
def test_detokenize_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    response = client.detokenize(
        model=None, checkpoint=checkpoint_name, token_ids=[4711, 42]
    )

    assert response["result"] is not None

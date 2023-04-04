import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.detokenization import DetokenizationRequest

from tests.common import (
    model_name,
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


# Client


@pytest.mark.system_test
def test_detokenize(sync_client: Client, model_name: str):
    response = sync_client.detokenize(DetokenizationRequest([4711]), model=model_name)

    assert response.result is not None

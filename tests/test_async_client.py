import os
import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt


@pytest.mark.needs_api
async def test_can_instantiate_async_client():
    token = os.environ.get("TEST_TOKEN")
    async with AsyncClient(token) as client:
        pass


@pytest.mark.needs_api
async def test_can_complete_with_async_client():
    token = os.environ.get("TEST_TOKEN")
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    async with AsyncClient(token) as client:
        response = await client.complete(request, model="luminous-base")

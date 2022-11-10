import os
import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt
from .common import model_name, checkpoint_name


@pytest.mark.needs_api
async def test_can_instantiate_async_client():
    token = os.environ.get("TEST_TOKEN")
    async with AsyncClient(token) as client:
        pass


@pytest.mark.needs_api
async def test_can_use_async_client_without_context_manager(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    client = await AsyncClient(token).open()
    _ = await client.complete(request, model=model_name)
    await client.close()


@pytest.mark.needs_api
async def test_can_complete_with_async_client(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    async with AsyncClient(token) as client:
        response = await client.complete(request, model=model_name)
        assert len(response.completions) == 1
        assert response.model_version is not None


@pytest.mark.needs_api
async def test_can_complete_with_async_client_against_checkpoint(checkpoint_name):
    token = os.environ.get("TEST_TOKEN")
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    async with AsyncClient(token) as client:
        response = await client.complete(request, checkpoint=checkpoint_name)
        assert len(response.completions) == 1
        assert response.model_version is not None

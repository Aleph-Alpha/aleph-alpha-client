import pytest
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.prompt import Prompt
from aleph_alpha_client.tokenization import TokenizationRequest

from tests.common import (
    sync_client,
    client,
    model_name,
    model,
    async_client,
)


# AsyncClient


@pytest.mark.system_test
async def test_can_tokenize_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)

    response = await async_client.tokenize(request, model=model_name)
    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


# Client


@pytest.mark.system_test
def test_tokenize(sync_client: Client, model_name: str):
    response = sync_client.tokenize(
        request=TokenizationRequest("Hello", tokens=True, token_ids=True),
        model=model_name,
    )

    assert response.tokens and len(response.tokens) == 1
    assert response.token_ids and len(response.token_ids) == 1


# ALephAlphaClient


@pytest.mark.system_test
def test_tokenize_with_client_against_model(client: AlephAlphaClient, model_name: str):
    response = client.tokenize(model_name, prompt="Hello", tokens=True, token_ids=True)

    assert len(response["tokens"]) == 1
    assert len(response["token_ids"]) == 1


def test_offline_tokenizer_sync(sync_client: Client, model_name: str):
    prompt = "Hello world"

    tokenizer = sync_client.offline_tokenizer(model_name)
    offline_tokenization = tokenizer.encode(prompt)
    print(offline_tokenization)

    request = TokenizationRequest(prompt=prompt, token_ids=True, tokens=True)
    response = sync_client.tokenize(request, model_name)
    print(response)

    assert offline_tokenization.ids == response.token_ids
    assert offline_tokenization.tokens == response.tokens

async def test_offline_tokenizer_async(async_client: AsyncClient, model_name: str):
    prompt = "Hello world"

    tokenizer = await async_client.offline_tokenizer(model_name)
    offline_tokenization = tokenizer.encode(prompt)
    print(offline_tokenization)

    request = TokenizationRequest(prompt=prompt, token_ids=True, tokens=True)
    response = await async_client.tokenize(request, model_name)
    print(response)

    assert offline_tokenization.ids == response.token_ids
    assert offline_tokenization.tokens == response.tokens

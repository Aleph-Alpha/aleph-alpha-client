import os
import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt
from aleph_alpha_client.detokenization import DetokenizationRequest
from aleph_alpha_client.tokenization import TokenizationRequest
from aleph_alpha_client.embedding import (
    EmbeddingRequest,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
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


@pytest.mark.needs_api
async def test_can_detokenization_with_async_client(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = DetokenizationRequest(token_ids=[2, 3, 4])
    async with AsyncClient(token) as client:
        response = await client.detokenize(request, model=model_name)
        assert len(response.result) > 0


@pytest.mark.needs_api
async def test_can_detokenization_with_async_client_with_checkpoint(checkpoint_name):
    token = os.environ.get("TEST_TOKEN")
    request = DetokenizationRequest(token_ids=[2, 3, 4])
    async with AsyncClient(token) as client:
        response = await client.detokenize(request, checkpoint=checkpoint_name)
        assert len(response.result) > 0


@pytest.mark.needs_api
async def test_can_tokenize_with_async_client(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)
    async with AsyncClient(token) as client:
        response = await client.tokenize(request, model=model_name)
        assert len(response.tokens) == 1
        assert len(response.token_ids) == 1


@pytest.mark.needs_api
async def test_can_tokenize_with_async_client_with_checkpoint(checkpoint_name):
    token = os.environ.get("TEST_TOKEN")
    request = TokenizationRequest(prompt="hello", token_ids=True, tokens=True)
    async with AsyncClient(token) as client:
        response = await client.tokenize(request, checkpoint=checkpoint_name)
        assert len(response.tokens) == 1
        assert len(response.token_ids) == 1


@pytest.mark.needs_api
async def test_can_embed_with_async_client(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[-1], pooling=["mean"], tokens=True
    )
    async with AsyncClient(token) as client:
        response = await client.embed(request, model=model_name)
        assert response.model_version is not None
        assert response.embeddings and len(response.embeddings) == len(
            request.pooling
        ) * len(request.layers)
        assert response.tokens is not None


@pytest.mark.needs_api
async def test_can_embed_with_async_client_against_checkpoint(checkpoint_name):
    token = os.environ.get("TEST_TOKEN")
    request = request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[-1], pooling=["mean"], tokens=True
    )
    async with AsyncClient(token) as client:
        response = await client.embed(request, checkpoint=checkpoint_name)
        assert response.model_version is not None
        assert response.embeddings and len(response.embeddings) == len(
            request.pooling
        ) * len(request.layers)
        assert response.tokens is not None


@pytest.mark.needs_api
async def test_can_semantic_embed_with_async_client(model_name):
    token = os.environ.get("TEST_TOKEN")
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )
    async with AsyncClient(token) as client:
        response = await client.semantic_embed(request, model=model_name)
        assert response.model_version is not None
        assert response.embedding
        assert len(response.embedding) == 128


@pytest.mark.needs_api
async def test_can_semantic_embed_with_async_client_against_checkpoint(checkpoint_name):
    token = os.environ.get("TEST_TOKEN")
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )
    async with AsyncClient(token) as client:
        response = await client.semantic_embed(request, checkpoint=checkpoint_name)
        assert response.model_version is not None
        assert response.embedding
        assert len(response.embedding) == 128
import json
import math
import random
from typing import Sequence
import pytest
from pytest_httpserver import HTTPServer

from aleph_alpha_client import EmbeddingRequest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.embedding import (
    BatchSemanticEmbeddingRequest,
    InstructableEmbeddingRequest,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
    BatchSemanticEmbeddingResponse,
)
from aleph_alpha_client.prompt import Prompt

# AsyncClient


async def test_can_embed_with_async_client(async_client: AsyncClient, model_name: str):
    request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[-1], pooling=["mean"], tokens=True
    )

    response = await async_client.embed(request, model=model_name)
    assert response.model_version is not None
    assert response.embeddings and len(response.embeddings) == len(
        request.pooling
    ) * len(request.layers)
    assert response.tokens is not None
    assert response.num_tokens_prompt_total >= 1


async def test_can_semantic_embed_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    response = await async_client.semantic_embed(request, model=model_name)
    assert response.model_version is not None
    assert response.embedding
    assert len(response.embedding) == 128
    assert response.num_tokens_prompt_total >= 1


@pytest.mark.parametrize("num_prompts", [1, 100, 101])
@pytest.mark.parametrize("batch_size", [1, 32, 100])
async def test_batch_embed_semantic_with_async_client(
    async_client: AsyncClient, sync_client: Client, num_prompts: int, batch_size: int
):
    words = ["car", "elephant", "kitchen sink", "rubber", "sun"]
    r = random.Random(4082)
    prompts = list(
        [Prompt.from_text(words[r.randint(0, 4)]) for i in range(num_prompts)]
    )

    request = BatchSemanticEmbeddingRequest(
        prompts=prompts,
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )
    result = await async_client.batch_semantic_embed(
        request=request,
        num_concurrent_requests=10,
        batch_size=batch_size,
        model="luminous-base",
    )

    # We have no control over the exact tokenizer used in the backend, so we cannot know the exact
    # number of tokens. We do know, however, that there must be at least one token per prompt.
    assert result.num_tokens_prompt_total >= num_prompts

    assert len(result.embeddings) == num_prompts
    # To make sure that the ordering of responses is preserved,
    # we compare the returned embeddings with those of the sync_client's
    # sequential implementation
    embeddings_approximately_equal(
        result.embeddings, sync_client.batch_semantic_embed(request=request).embeddings
    )


@pytest.mark.parametrize("batch_size", [-1, 0, 101])
async def test_batch_embed_semantic_invalid_batch_sizes(
    async_client: AsyncClient, batch_size: int
):
    words = ["car", "elephant", "kitchen sink", "rubber", "sun"]
    request = BatchSemanticEmbeddingRequest(
        prompts=[Prompt.from_text(word) for word in words],
        representation=SemanticRepresentation.Symmetric,
    )

    with pytest.raises(ValueError):
        await async_client.batch_semantic_embed(request=request, batch_size=batch_size)


async def test_can_instructable_embed_with_async_client(
    async_client: AsyncClient,
):
    request = InstructableEmbeddingRequest(
        input=Prompt.from_text("hello"),
        instruction="Represent the text to query a database of salutations",
    )

    response = await async_client.instructable_embed(
        request, model="pharia-1-embedding-4608-control"
    )
    assert response.model_version is not None
    assert response.embedding
    assert len(response.embedding) == 4608
    assert response.num_tokens_prompt_total >= 1


def cosine_similarity(emb1: Sequence[float], emb2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0.0, 0.0, 0.0
    for i in range(len(emb1)):
        x = emb1[i]
        y = emb2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def embeddings_approximately_equal(a, b):
    assert all([cosine_similarity(v1, v2) > 0.99 for (v1, v2) in zip(a, b)])


async def test_modelname_gets_passed_along_for_async_client(httpserver: HTTPServer):
    request = BatchSemanticEmbeddingRequest(
        prompts=[Prompt.from_text("hello")],
        representation=SemanticRepresentation.Symmetric,
    )
    model_name = "test_model"
    expected_body = {
        **request.to_json(),
        "model": model_name,
    }
    httpserver.expect_ordered_request(
        "/batch_semantic_embed", method="POST", data=json.dumps(expected_body)
    ).respond_with_json(
        BatchSemanticEmbeddingResponse(
            model_version="1", embeddings=[], num_tokens_prompt_total=1
        ).to_json()
    )
    async_client = AsyncClient(token="", host=httpserver.url_for(""), total_retries=1)
    await async_client.batch_semantic_embed(request, model=model_name)


# Client


def test_embed(sync_client: Client, model_name: str):
    request = EmbeddingRequest(
        prompt=Prompt.from_text("hello"), layers=[0, -1], pooling=["mean", "max"]
    )

    result = sync_client.embed(request=request, model=model_name)

    assert result.model_version is not None
    assert result.embeddings and len(result.embeddings) == len(request.pooling) * len(
        request.layers
    )
    assert result.tokens is None
    assert result.num_tokens_prompt_total >= 1


def test_embedding_of_one_token_aggregates_identically(
    sync_client: Client, model_name: str
):
    request = EmbeddingRequest(
        prompt=Prompt.from_text(
            "hello"
        ),  # it is important for this test that we only embed one single token
        layers=[0, -1],
        pooling=["mean", "max"],
    )

    result = sync_client.embed(request, model=model_name)

    assert (
        result.embeddings
        and result.embeddings[("layer_0", "mean")]
        == result.embeddings[("layer_0", "max")]
    )


def test_embed_with_tokens(sync_client: Client, model_name: str):
    request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[-1], pooling=["mean"], tokens=True
    )

    result = sync_client.embed(request, model=model_name)

    assert result.model_version is not None
    assert result.embeddings and len(result.embeddings) == len(request.pooling) * len(
        request.layers
    )
    assert result.tokens is not None
    assert result.num_tokens_prompt_total >= 1


def test_embed_semantic(sync_client: Client):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    result = sync_client.semantic_embed(request=request, model="luminous-base")

    assert result.model_version is not None
    assert result.embedding
    assert len(result.embedding) == 128
    assert result.num_tokens_prompt_total >= 1


def test_embed_instructable(sync_client: Client):
    request = InstructableEmbeddingRequest(
        input=Prompt.from_text("hello"),
        instruction="Represent the text to query a database of salutations",
    )

    result = sync_client.instructable_embed(
        request=request, model="pharia-1-embedding-4608-control"
    )

    assert result.model_version is not None
    assert result.embedding
    assert len(result.embedding) == 4608
    assert result.num_tokens_prompt_total >= 1


@pytest.mark.parametrize("num_prompts", [1, 100, 101, 200])
def test_batch_embed_semantic(sync_client: Client, num_prompts: int):
    request = BatchSemanticEmbeddingRequest(
        prompts=[Prompt.from_text("hello") for _ in range(num_prompts)],
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    result = sync_client.batch_semantic_embed(request=request, model="luminous-base")
    assert len(result.embeddings) == num_prompts


def test_modelname_gets_passed_along_for_sync_client(httpserver: HTTPServer):
    request = BatchSemanticEmbeddingRequest(
        prompts=[Prompt.from_text("hello")],
        representation=SemanticRepresentation.Symmetric,
    )
    model_name = "test_model"
    expected_body = {**request.to_json(), "model": model_name}
    httpserver.expect_ordered_request(
        "/batch_semantic_embed", method="POST", data=json.dumps(expected_body)
    ).respond_with_json(
        BatchSemanticEmbeddingResponse(
            model_version="1", embeddings=[], num_tokens_prompt_total=1
        ).to_json()
    )
    sync_client = Client(token="", host=httpserver.url_for(""), total_retries=1)
    sync_client.batch_semantic_embed(request, model=model_name)

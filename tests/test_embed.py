from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient, EmbeddingRequest
from tests.common import client, model


def test_embed(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt=["hello"],
        layers=[0, -1],
        pooling=["mean", "max"])

    result = client.embed(model=model, request=request)

    assert result.model_version is not None
    assert len(result.embeddings) == len(request.pooling) * len(request.layers)
    assert result.tokens is None


def test_embedding_of_one_token_aggregates_identically(client: AlephAlphaClient, model: str):
    request = EmbeddingRequest(
        prompt=["hello"], # it is important for this test that we only embed one single token
        layers=[0, -1],
        pooling=["mean", "max"])

    result = client.embed(model=model, request=request)

    assert result.embeddings[("layer_0", "mean")] == result.embeddings[("layer_0", "max")]


def test_embed_with_tokens(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt=["abc"],
        layers=[-1],
        pooling=["mean"],
        tokens=True)

    result = client.embed(model=model, request=request)

    assert result.model_version is not None
    assert len(result.embeddings) == len(request.pooling) * len(request.layers)
    assert result.tokens is not None


def test_failing_embedding_request(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt=["abc"],
        layers=[0, 1, 2],
        pooling=["mean"])

    with pytest.raises(ValueError):
        client.embed(model=model, request=request)
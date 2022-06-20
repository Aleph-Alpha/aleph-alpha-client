import pytest
from aleph_alpha_client import AlephAlphaClient, EmbeddingRequest
from tests.common import client, model


def test_embed(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt="abc",
        layers=[0, -1],
        pooling=["mean", "max"],
        type=None)

    result = client.embed(model=model, request=request)

    assert result.model_version is not None
    assert len(result.embeddings) == len(request.pooling) * len(request.layers)
    assert result.tokens is None


def test_embed_with_tokens(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt="abc",
        layers=[-1],
        pooling=["mean"],
        type=None,
        tokens=True)

    result = client.embed(model=model, request=request)

    assert result.model_version is not None
    assert len(result.embeddings) == len(request.pooling) * len(request.layers)
    assert result.tokens is not None

def test_failing_embedding_request(client: AlephAlphaClient, model: str):

    request = EmbeddingRequest(
        prompt="abc",
        layers=[0, 1, 2],
        pooling=["mean"],
        type=None,
        tokens=True)

    with pytest.raises(ValueError):
        client.embed(model=model, request=request)
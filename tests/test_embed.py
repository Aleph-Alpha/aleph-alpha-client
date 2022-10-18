from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient, EmbeddingRequest
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.embedding import (
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from aleph_alpha_client.prompt import Prompt
from tests.common import client, checkpoint_name, model_name, luminous_base, model


@pytest.mark.needs_api
def test_embed(model: AlephAlphaModel):

    request = EmbeddingRequest(
        prompt=Prompt.from_text("hello"), layers=[0, -1], pooling=["mean", "max"]
    )

    result = model.embed(request=request)

    assert result.model_version is not None
    assert result.embeddings and len(result.embeddings) == len(request.pooling) * len(
        request.layers
    )
    assert result.tokens is None


@pytest.mark.needs_api
def test_embed_against_checkpoint(client: AlephAlphaClient, checkpoint_name: str):
    model = AlephAlphaModel(client, checkpoint_name=checkpoint_name)

    request = EmbeddingRequest(
        prompt=Prompt.from_text("hello"), layers=[0, -1], pooling=["mean", "max"]
    )

    result = model.embed(request=request)

    assert result.model_version is not None
    assert result.embeddings and len(result.embeddings) == len(request.pooling) * len(
        request.layers
    )
    assert result.tokens is None


@pytest.mark.needs_api
def test_embed_with_client(client: AlephAlphaClient, model_name: str):
    layers = [0, -1]
    pooling = ["mean", "max"]
    prompt = ["hello"]

    result = client.embed(model_name, prompt, pooling, layers)

    assert result["model_version"] is not None
    assert len(result["embeddings"]) == len(layers)
    assert len(result["embeddings"]["layer_0"]) == len(pooling)
    assert result["tokens"] is None


@pytest.mark.needs_api
def test_embed_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    layers = [0, -1]
    pooling = ["mean", "max"]
    prompt = ["hello"]

    result = client.embed(
        model=None,
        prompt=prompt,
        pooling=pooling,
        layers=layers,
        checkpoint=checkpoint_name,
    )

    assert result["model_version"] is not None
    assert len(result["embeddings"]) == len(layers)
    assert len(result["embeddings"]["layer_0"]) == len(pooling)
    assert result["tokens"] is None


@pytest.mark.needs_api
def test_embedding_of_one_token_aggregates_identically(model: AlephAlphaModel):
    request = EmbeddingRequest(
        prompt=Prompt.from_text(
            "hello"
        ),  # it is important for this test that we only embed one single token
        layers=[0, -1],
        pooling=["mean", "max"],
    )

    result = model.embed(request)

    assert (
        result.embeddings
        and result.embeddings[("layer_0", "mean")]
        == result.embeddings[("layer_0", "max")]
    )


@pytest.mark.needs_api
def test_embed_with_tokens(model: AlephAlphaModel):
    request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[-1], pooling=["mean"], tokens=True
    )

    result = model.embed(request)

    assert result.model_version is not None
    assert result.embeddings and len(result.embeddings) == len(request.pooling) * len(
        request.layers
    )
    assert result.tokens is not None


@pytest.mark.needs_api
def test_failing_embedding_request(model: AlephAlphaModel):
    # given a client
    assert model.model_name in (
        model["name"] for model in model.client.available_models()
    )

    # when posting an illegal request
    request = EmbeddingRequest(
        prompt=Prompt.from_text("abc"), layers=[0, 1, 2], pooling=["mean"]
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        model.embed(request)

    assert e.value.args[0] == 400


@pytest.mark.needs_api
def test_embed_semantic(luminous_base: AlephAlphaModel):

    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    result = luminous_base.semantic_embed(request=request)

    assert result.model_version is not None
    assert result.embedding
    assert len(result.embedding) == 128


@pytest.mark.needs_api
def test_embed_semantic_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    model = AlephAlphaModel(client, checkpoint_name=checkpoint_name)

    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    result = model.semantic_embed(request=request)

    assert result.model_version is not None
    assert result.embedding
    assert len(result.embedding) == 128


@pytest.mark.needs_api
def test_embed_semantic_client(client: AlephAlphaClient):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )
    result = client.semantic_embed(
        model="luminous-base",
        request=request,
    )
    # result = luminous_base.semantic_embed(request=request)

    assert result["model_version"] is not None
    assert result["embedding"]
    assert len(result["embedding"]) == 128


@pytest.mark.needs_api
def test_semantic_embed_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text("hello"),
        representation=SemanticRepresentation.Symmetric,
        compress_to_size=128,
    )

    result = client.semantic_embed(
        model=None,
        request=request,
        checkpoint=checkpoint_name,
    )

    assert result["model_version"] is not None
    assert result["embedding"]
    assert len(result["embedding"]) == 128

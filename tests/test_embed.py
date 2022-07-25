from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient, EmbeddingRequest
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.embedding import EmbeddingForSearchRequest
from aleph_alpha_client.prompt import Prompt
from tests.common import client, model_name, luminous_base, model


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


def test_embed_with_client(client: AlephAlphaClient, model_name: str):
    layers = [0, -1]
    pooling = ["mean", "max"]
    prompt = ["hello"]

    result = client.embed(model_name, prompt, pooling, layers)

    assert result["model_version"] is not None
    assert len(result["embeddings"]) == len(layers)
    assert len(result["embeddings"]["layer_0"]) == len(pooling)
    assert result["tokens"] is None


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


def test_embed_semantic(luminous_base: AlephAlphaModel):

    request = EmbeddingForSearchRequest(
        prompt=Prompt.from_text("hello"),
        type="symmetric",
    )

    result = luminous_base.embed_for_search(request=request)

    assert result.model_version is not None
    assert result.embeddings
    assert len(result.embeddings) == 1
    assert ("symmetric", "weighted_mean") in result.embeddings
    assert result.tokens is None

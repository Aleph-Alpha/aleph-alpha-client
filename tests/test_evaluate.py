from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.evaluation import EvaluationRequest
from tests.common import client, model


def test_evaluate(client: AlephAlphaClient, model: str):

    request = EvaluationRequest(prompt=["hello"], completion_expected="world")

    result = client.evaluate(model=model, request=request)

    assert result.model_version is not None
    assert result.result is not None


def test_embed_with_explicit_parameters(client: AlephAlphaClient, model: str):
    result = client.evaluate(model, prompt="hello", completion_expected="world")

    assert result["model_version"] is not None
    assert result["result"] is not None

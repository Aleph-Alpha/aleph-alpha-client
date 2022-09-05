from multiprocessing.sharedctypes import Value
from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.evaluation import EvaluationRequest
from aleph_alpha_client.prompt import Prompt
from tests.common import client, model_name, model, checkpoint_name


def test_evaluate(model: AlephAlphaModel):

    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"), completion_expected="world"
    )

    result = model.evaluate(request)

    assert result.model_version is not None
    assert result.result is not None


def test_evaluate_with_client(client: AlephAlphaClient, model_name: str):
    result = client.evaluate(model_name, prompt="hello", completion_expected="world")

    assert result["model_version"] is not None
    assert result["result"] is not None


def test_evaluate_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):
    result = client.evaluate(
        model=None,
        prompt="hello",
        completion_expected="world",
        checkpoint=checkpoint_name,
    )

    assert result["model_version"] is not None
    assert result["result"] is not None


def test_evaluate_fails(model: AlephAlphaModel):
    # given a client
    assert model.model_name in map(
        lambda model: model["name"], model.client.available_models()
    )

    # when posting an illegal request
    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"),
        completion_expected="",
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = model.evaluate(request=request)

    assert e.value.args[0] == 400

from multiprocessing.sharedctypes import Value
from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.evaluation import EvaluationRequest
from tests.common import client, model_name


def test_evaluate(client: AlephAlphaClient, model_name: str):

    request = EvaluationRequest(prompt=["hello"], completion_expected="world")

    result = client.evaluate(model=model_name, request=request)

    assert result.model_version is not None
    assert result.result is not None


def test_evaluate_with_explicit_parameters(client: AlephAlphaClient, model_name: str):
    result = client.evaluate(model_name, prompt="hello", completion_expected="world")

    assert result["model_version"] is not None
    assert result["result"] is not None


def test_evaluate_fails(client: AlephAlphaClient, model_name: str):
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting an illegal request
    request = EvaluationRequest(
        prompt=["hello"],
        completion_expected="",
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = client.evaluate(
            model_name,
            hosting="cloud",
            request=request,
        )

    assert e.value.args[0] == 400

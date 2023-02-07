from multiprocessing.sharedctypes import Value
from typing import List
import pytest
from aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.evaluation import EvaluationRequest
from aleph_alpha_client.prompt import Prompt
from tests.common import (
    sync_client,
    client,
    model_name,
    model,
    async_client,
)


# AsyncClient


@pytest.mark.system_test
async def test_can_evaluate_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"), completion_expected="world"
    )

    response = await async_client.evaluate(request, model=model_name)
    assert response.model_version is not None
    assert response.result is not None


# Client


@pytest.mark.system_test
def test_evaluate(sync_client: Client, model_name: str):

    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"), completion_expected="world"
    )

    result = sync_client.evaluate(request, model=model_name)

    assert result.model_version is not None
    assert result.result is not None


# AlephAlphaClient


@pytest.mark.system_test
def test_evaluate_with_client(client: AlephAlphaClient, model_name: str):
    result = client.evaluate(model_name, prompt="hello", completion_expected="world")

    assert result["model_version"] is not None
    assert result["result"] is not None

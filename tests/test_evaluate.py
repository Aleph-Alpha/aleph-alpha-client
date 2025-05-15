import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.evaluation import EvaluationRequest
from aleph_alpha_client.prompt import Prompt


# AsyncClient


async def test_can_evaluate_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"), completion_expected="world"
    )

    response = await async_client.evaluate(request, model=model_name)
    assert response.model_version is not None
    assert response.result is not None
    assert response.num_tokens_prompt_total >= 1


# Client


def test_evaluate(sync_client: Client, model_name: str):
    request = EvaluationRequest(
        prompt=Prompt.from_text("hello"), completion_expected="world"
    )

    response = sync_client.evaluate(request, model=model_name)

    assert response.model_version is not None
    assert response.result is not None
    assert response.num_tokens_prompt_total >= 1

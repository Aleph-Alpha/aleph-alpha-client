import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.explanation import Explanation2Request
from aleph_alpha_client.prompt import Prompt

from tests.common import (
    sync_client,
    client,
    model_name,
    model,
    async_client,
)


# AsynClient


async def test_can_explain_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )

    response = await async_client._explain(request, model=model_name)
    assert response.result


# Client


def test_explanation(sync_client: Client, model_name: str):
    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )

    explanation = sync_client._explain(request, model=model_name)

    assert len(explanation.result) > 0


def test_explanation2(sync_client: Client, model_name: str):
    request = Explanation2Request(
        prompt=Prompt.from_text("I am a programmer and French. My favourite food is"),
        target=" pizza with cheese",
    )

    explanation = sync_client._explain2(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 2 for exp in explanation.explanations])

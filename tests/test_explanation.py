import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
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


# AlephAlphaClient

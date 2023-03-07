import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.explanation import ExplanationRequest
from aleph_alpha_client.prompt import Prompt, Text

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
        prompt=Prompt(
            [
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                [4014, 36316, 5681, 387],
            ]
        ),
        target=" pizza with cheese",
    )

    explanation = await async_client._explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 3 for exp in explanation.explanations])


# Client


def test_explanation(sync_client: Client, model_name: str):
    request = ExplanationRequest(
        prompt=Prompt(
            [
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                [4014, 36316, 5681, 387],
            ]
        ),
        target=" pizza with cheese",
    )

    explanation = sync_client._explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 3 for exp in explanation.explanations])

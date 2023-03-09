from pathlib import Path
import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client import ExplanationGranularity, ExplanationRequest
from aleph_alpha_client import Image
from aleph_alpha_client import Prompt, Text
from aleph_alpha_client.explanation import ExplanationPostprocessing

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
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    img = Image.from_image_source(image_source=str(image_source_path))

    request = ExplanationRequest(
        prompt=Prompt(
            [
                img,
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                [4014, 36316, 5681, 387],
            ]
        ),
        target=" pizza with cheese",
        granularity=ExplanationGranularity.Word,
        postprocessing=ExplanationPostprocessing.Absolute,
        normalize=True,
    )

    explanation = sync_client._explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 3 for exp in explanation.explanations])
    # At least one of the following options must be set in the request
    # to make all scores positive (or zero):
    # postprocessing=ExplanationPostProcessing.Absolute
    # postprocessing=ExplanationPostProcessing.Square
    # normalize=true
    for exp in explanation.explanations:
        for prompt_item in exp.items:
            assert all([score.score >= 0.0 for score in prompt_item.scores])

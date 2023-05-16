from pathlib import Path
from aleph_alpha_client.explanation import (
    TargetScoreWithRaw,
    TextScoreWithRaw,
)
from aleph_alpha_client.prompt import Tokens
import pytest
from aleph_alpha_client import (
    ControlTokenOverlap,
    ExplanationRequest,
    AsyncClient,
    Client,
    ExplanationRequest,
    Image,
    Prompt,
    Text,
    CustomGranularity,
    TargetGranularity,
    ExplanationPostprocessing,
    ImageScore,
)

from tests.common import (
    sync_client,
    model_name,
    async_client,
)


# AsynClient


async def test_can_explain_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = ExplanationRequest(
        prompt=Prompt(
            [
                Text.from_text(
                    "I am a programmer###I am French###I don't like pizza###My favourite food is"
                ),
                # " My favorite food is"
                Tokens.from_token_ids([4014, 36316, 5681, 387]),
            ]
        ),
        target=" pizza with cheese",
        prompt_granularity=CustomGranularity("###"),
        target_granularity=TargetGranularity.Token,
    )

    explanation = await async_client.explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 3 for exp in explanation.explanations])
    assert len(explanation.explanations[0].items[0].scores) == 4


# Client

@pytest.mark.system_test
def test_explanation(sync_client: Client, model_name: str):
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    img = Image.from_image_source(image_source=str(image_source_path))

    request = ExplanationRequest(
        prompt=Prompt(
            [
                img,
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                Tokens.from_token_ids([4014, 36316, 5681, 387]),
            ]
        ),
        target=" pizza with cheese",
        prompt_granularity="sentence",
        postprocessing=ExplanationPostprocessing.Absolute,
        normalize=True,
        target_granularity=TargetGranularity.Token,
        control_token_overlap=ControlTokenOverlap.Complete,
    )

    explanation = sync_client.explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 4 for exp in explanation.explanations])
    # At least one of the following options must be set in the request
    # to make all scores positive (or zero):
    # postprocessing=ExplanationPostProcessing.Absolute
    # postprocessing=ExplanationPostProcessing.Square
    # normalize=true
    for exp in explanation.explanations:
        for prompt_item in exp.items:
            assert all([score.score >= 0.0 for score in prompt_item.scores])


def test_explanation_auto_granularity(sync_client: Client, model_name: str):
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    img = Image.from_image_source(image_source=str(image_source_path))

    request = ExplanationRequest(
        prompt=Prompt(
            [
                img,
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                Tokens.from_token_ids([4014, 36316, 5681, 387]),
            ]
        ),
        target=" pizza with cheese",
        prompt_granularity=None,
    )

    explanation = sync_client.explain(request, model=model_name)

    assert len(explanation.explanations) == 1
    assert all([len(exp.items) == 4 for exp in explanation.explanations])


def test_explanation_of_image_in_pixels(sync_client: Client, model_name: str):
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    img = Image.from_image_source(image_source=str(image_source_path))

    request = ExplanationRequest(
        prompt=Prompt(
            [
                img,
                Text.from_text("I am a programmer and French. My favourite food is"),
                # " My favorite food is"
                Tokens.from_token_ids([4014, 36316, 5681, 387]),
            ]
        ),
        target=" pizza with cheese",
        prompt_granularity=None,
    )

    explanation = sync_client.explain(request, model=model_name)

    explanation = explanation.with_image_prompt_items_in_pixels(request.prompt)
    assert len(explanation.explanations) == 1
    assert all([len(exp.items) == 4 for exp in explanation.explanations])
    assert all(
        [
            isinstance(image_score, ImageScore) and isinstance(image_score.left, int)
            for image_score in explanation.explanations[0].items[0].scores
        ]
    )


def test_explanation_with_text_from_request(sync_client: Client, model_name: str):
    request = ExplanationRequest(
        prompt=Prompt(
            [
                Text.from_text("I am a programmer and French. My favourite food is"),
                Tokens.from_token_ids([4014, 36316, 5681, 387]),
            ]
        ),
        target=" pizza with cheese",
        prompt_granularity=None,
        target_granularity=TargetGranularity.Token,
    )

    explanation = sync_client.explain(request, model=model_name)

    explanation = explanation.with_text_from_prompt(request)
    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 3 for exp in explanation.explanations])
    assert all(
        [
            isinstance(raw_text_score, TextScoreWithRaw)
            and isinstance(raw_text_score.text, str)
            for raw_text_score in explanation.explanations[0].items[0].scores
        ]
    )
    assert all(
        [
            isinstance(raw_text_score, TargetScoreWithRaw)
            and isinstance(raw_text_score.text, str)
            for raw_text_score in explanation.explanations[1].items[2].scores
        ]
    )

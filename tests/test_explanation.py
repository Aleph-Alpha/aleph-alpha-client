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
    Image,
    Prompt,
    Text,
    CustomGranularity,
    TargetGranularity,
    PromptGranularity,
    ExplanationPostprocessing,
    ImageScore,
)


# AsyncClient


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


def test_explanation_with_text_only(sync_client: Client, model_name: str):
    request = ExplanationRequest(
        prompt=Prompt.from_text("I am a programmer and French. My favourite food is"),
        target=" pizza with cheese",
        target_granularity=TargetGranularity.Token,
        normalize=True,
    )

    explanation = sync_client.explain(request, model=model_name)

    assert len(explanation.explanations) == 3
    assert all([len(exp.items) == 2 for exp in explanation.explanations])
    # At least one of the following options must be set in the request
    # to make all scores positive (or zero):
    # postprocessing=ExplanationPostProcessing.Absolute
    # postprocessing=ExplanationPostProcessing.Square
    # normalize=true
    for exp in explanation.explanations:
        for prompt_item in exp.items:
            assert all([score.score >= 0.0 for score in prompt_item.scores])


def test_explanation_with_multimodal_prompt(sync_client: Client, model_name: str):
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


# Regression test for out-of-memory errors that could be triggered on the workers with the explanation job shown below
def test_explanation_with_token_granularities_oom_regression(sync_client: Client):
    prompt_text = """### Instruction:
Answer the question using the Source. If there's no answer, say "NO_ANSWER_IN_TEXT".

### Input:
Source: The Battle of Waterloo was fought on Sunday 18 June 1815, near Waterloo (at that time in the United Kingdom of the Netherlands, now in Belgium). A French army under the command of Napoleon was defeated by two of the armies of the Seventh Coalition. One of these was a British-led coalition consisting of units from the United Kingdom, the Netherlands, Hanover, Brunswick, and Nassau, under the command of the Duke of Wellington (referred to by many authors as the Anglo-allied army or Wellington's army). The other was composed of three corps of the Prussian army under the command of Field Marshal von Blücher (the fourth corps of this army fought at the Battle of Wavre on the same day). The battle marked the end of the Napoleonic Wars. The battle was contemporaneously known as the Battle of Mont Saint-Jean (France) or La Belle Alliance ("the Beautiful Alliance" – Prussia).
Question: Did India win or lost the Battle of Waterloo?

### Response:"""
    target_text = " India won the Battle of Waterloo."
    model_name = "luminous-base"
    request = ExplanationRequest(
        prompt=Prompt([Text(prompt_text, controls=[])]),
        target=target_text,
        prompt_granularity=PromptGranularity.Token,
        target_granularity=TargetGranularity.Token,
        control_factor=0.1,
        control_log_additive=True,
        postprocessing=None,
        control_token_overlap=ControlTokenOverlap.Partial,
        contextual_control_threshold=None,
    )
    explanation = sync_client.explain(request, model=model_name)
    assert len(explanation.explanations) == 7

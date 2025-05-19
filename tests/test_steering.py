import pytest
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.steering import (
    SteeringConceptCreationRequest,
    SteeringPairedExample,
)

from .conftest import GenericClient, llama_prompt, requires_beta_features


@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
@requires_beta_features
async def test_can_create_and_use_steering_concept(
    generic_client: GenericClient, chat_model_name: str
):
    # 1. We create a new steering concept with examples that steer the models
    # output away from formal english towards slang.
    response = await generic_client.create_steering_concept(
        create_sample_steering_concept_creation_request()
    )
    steering_concept_id = response.id
    assert isinstance(steering_concept_id, str) and len(steering_concept_id) > 0

    # 2. We ask the model to paraphrase "You are an honest man.", with and
    # without the steering concept applied.
    async def complete(**kwargs) -> str:
        response = await generic_client.complete(
            CompletionRequest(
                prompt=llama_prompt(
                    "Reply only with a paraphrased version of the following phrase: "
                    "You are an honest man."
                ),
                maximum_tokens=16,
                **kwargs,
            ),
            model=chat_model_name,
        )
        assert isinstance(response.completions[0].completion, str)
        return response.completions[0].completion

    base_completion = await complete()
    steered_completion = await complete(steering_concepts=[steering_concept_id])

    # The outputs for my test runs were:
    # base_completion == "Your integrity is evident in all that you do."
    # steered_completion == "You're a straight shooter, no games."
    #
    # In order to reduce the flakiness of tests we don't assert exact model
    # outputs.
    assert isinstance(base_completion, str)
    assert isinstance(steered_completion, str)
    assert base_completion != steered_completion


def create_sample_steering_concept_creation_request() -> SteeringConceptCreationRequest:
    return SteeringConceptCreationRequest(
        examples=[
            SteeringPairedExample(
                negative="I appreciate your valuable feedback on this matter.",
                positive="Thanks for the real talk, fam.",
            ),
            SteeringPairedExample(
                negative="The financial projections indicate significant growth potential.",
                positive="Yo, these numbers are looking mad stacked!",
            ),
        ]
    )

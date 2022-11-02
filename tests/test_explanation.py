import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.prompt import Prompt

from tests.common import client, model_name, model, checkpoint_name


@pytest.mark.needs_api
def test_explanation(model: AlephAlphaModel):
    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )

    explanation = model._explain(request)

    assert len(explanation["result"]) > 0


@pytest.mark.needs_api
def test_explanation_against_checkpoint(client: AlephAlphaClient, checkpoint_name: str):
    model = AlephAlphaModel(client, checkpoint_name=checkpoint_name)
    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )
    explanation = model._explain(request)

    assert len(explanation["result"]) > 0


@pytest.mark.needs_api
def test_explanation_with_client_against_checkpoint(
    client: AlephAlphaClient, checkpoint_name: str
):

    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )

    explanation = client._explain(
        model=None, request=request, checkpoint=checkpoint_name
    )

    # List is true if not None and not empty
    assert explanation["result"]

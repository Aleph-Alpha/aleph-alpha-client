import pytest
from aleph_alpha_client import ExplanationRequest, AlephAlphaClient
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.prompt import Prompt

from tests.common import client, model_name, model, checkpoint_name


def test_explanation(model: AlephAlphaModel):

    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
    )

    explanation = model._explain(request)

    # List is true if not None and not empty
    assert explanation["result"]


def test_explain_fails(model: AlephAlphaModel):
    # given a client
    assert model.model_name in map(
        lambda model: model["name"], model.client.available_models()
    )

    # when posting an illegal request
    request = ExplanationRequest(
        prompt=Prompt.from_text("An apple a day"),
        target=" keeps the doctor away",
        suppression_factor=0.1,
        conceptual_suppression_threshold=-1,
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = model._explain(request)

    assert e.value.args[0] == 400


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

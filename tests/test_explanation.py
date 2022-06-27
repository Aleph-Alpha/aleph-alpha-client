import pytest
from aleph_alpha_client import AlephAlphaClient, ExplanationRequest

from tests.common import client, model_name


def test_explanation(client: AlephAlphaClient, model_name: str):

    request = ExplanationRequest(
        prompt=["An apple a day"],
        target=" keeps the doctor away",
        directional=False,
        suppression_factor=0.1,
    )

    explanation = client._explain(model=model_name, request=request, hosting=None)

    # List is true if not None and not empty
    assert explanation["result"]


def test_explain_fails(client: AlephAlphaClient, model_name: str):
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting an illegal request
    request = ExplanationRequest(
        prompt=["An apple a day"],
        target=" keeps the doctor away",
        directional=False,
        suppression_factor=0.1,
        conceptual_suppression_threshold=-1,
    )

    # then we expect an exception tue to a bad request response from the API
    with pytest.raises(ValueError) as e:
        response = client._explain(
            model_name,
            hosting="cloud",
            request=request,
        )

    assert e.value.args[0] == 400

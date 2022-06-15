from aleph_alpha_client import AlephAlphaClient, ExplanationRequest

from tests.common import client, model


def test_explanation(client: AlephAlphaClient, model: str):

    request = ExplanationRequest(
        prompt=["An apple a day"],
        target=" keeps the doctor away",
        directional=False,
        suppression_factor=0.1,
    )

    explanation = client._explain(model=model, request=request, hosting=None)

    # List is true if not None and not empty
    assert explanation["result"]

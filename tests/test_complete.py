from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.completion import CompletionRequest

from tests.common import client, model


def test_complete(client: AlephAlphaClient, model: str):
    response = client.complete(
        model,
        hosting="cloud",
        request=CompletionRequest(
            prompt="", maximum_tokens=7, tokens=False, log_probs=0
        ),
    )

    assert len(response.completions) == 1
    assert response.model_version is not None


def test_complete_with_explicit_parameters(client: AlephAlphaClient, model: str):
    response = client.complete(
        model, prompt="", maximum_tokens=7, tokens=False, log_probs=0
    )

    print(response)
    assert len(response["completions"]) == 1
    assert response["model_version"] is not None

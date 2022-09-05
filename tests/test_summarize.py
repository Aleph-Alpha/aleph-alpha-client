from aleph_alpha_client import (
    AlephAlphaClient,
    AlephAlphaModel,
    Document,
    SummarizationRequest,
)

from tests.common import (
    client,
    model_name,
    luminous_extended,
    summarization_checkpoint_name,
)


def test_summarize(luminous_extended: AlephAlphaModel):
    # given a client
    assert luminous_extended.model_name in map(
        lambda model: model["name"], luminous_extended.client.available_models()
    )

    # when posting a Summarization request
    request = SummarizationRequest(
        document=Document.from_prompt(["Andreas likes pizza."]),
    )

    response = luminous_extended.summarize(request)

    # the response should exist and be in the form of a named tuple class
    assert response.summary is not None
    assert response.model_version is not None


def test_summarization_with_client(client: AlephAlphaClient):
    model_name = "luminous-extended"
    # given a client
    assert model_name in map(lambda model: model["name"], client.available_models())

    # when posting a Summarization request
    response = client.summarize(
        "luminous-extended",
        SummarizationRequest(
            document=Document.from_prompt(["Andreas likes pizza."]),
        ),
    )

    # The response should exist in the form of a json dict
    assert response["summary"] is not None
    assert response["model_version"] is not None


def test_summarization_with_client_against_checkpoint(
    client: AlephAlphaClient, summarization_checkpoint_name
):
    # when posting a Summarization request
    response = client.summarize(
        model=None,
        request=SummarizationRequest(
            document=Document.from_prompt(["Andreas likes pizza."]),
        ),
        checkpoint=summarization_checkpoint_name,
    )

    # The response should exist in the form of a json dict
    assert response["summary"] is not None
    assert response["model_version"] is not None


def test_text(luminous_extended: AlephAlphaModel):
    # given a client
    assert luminous_extended.model_name in map(
        lambda model: model["name"], luminous_extended.client.available_models()
    )

    request = SummarizationRequest(
        document=Document.from_text("Andreas likes pizza."),
    )

    response = luminous_extended.summarize(request)

    assert response.summary is not None
    assert response.model_version is not None

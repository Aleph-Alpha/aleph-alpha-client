from pytest_httpserver import HTTPServer
import os
import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionResult,
)
from aleph_alpha_client.prompt import Prompt
from tests.common import model_name, sync_client, async_client


@pytest.mark.system_test
async def test_can_use_async_client_without_context_manager(model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    token = os.environ["TEST_TOKEN"]
    client = AsyncClient(token, host=os.environ["TEST_API_URL"])
    try:
        _ = await client.complete(request, model=model_name)
    finally:
        await client.close()


def test_nice_flag_on_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data("OK")

    httpserver.expect_request(
        "/complete", query_string={"nice": "true"}
    ).respond_with_json(
        CompletionResponse(
            "model_version",
            [CompletionResult(log_probs=[], completion="foo")],
        ).to_json()
    )

    client = Client(host=httpserver.url_for(""), token="AA_TOKEN", nice=True)

    request = CompletionRequest(prompt=Prompt.from_text("Hello world"))
    client.complete(request, model="luminous")


async def test_nice_flag_on_async_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data("OK")

    httpserver.expect_request(
        "/complete", query_string={"nice": "true"}
    ).respond_with_json(
        CompletionResponse(
            "model_version",
            [CompletionResult(log_probs=[], completion="foo")],
        ).to_json()
    )

    request = CompletionRequest(prompt=Prompt.from_text("Hello world"))

    async with AsyncClient(
        host=httpserver.url_for(""), token="AA_TOKEN", nice=True
    ) as client:
        await client.complete(request, model="luminous")


@pytest.mark.system_test
def test_available_models_sync_client(sync_client: Client, model_name: str):
    models = sync_client.models()
    assert model_name in {model["name"] for model in models}


@pytest.mark.system_test
async def test_available_models_async_client(
    async_client: AsyncClient, model_name: str
):
    models = await async_client.models()
    assert model_name in {model["name"] for model in models}

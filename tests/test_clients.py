from pytest_httpserver import HTTPServer
import os
import pytest

from aleph_alpha_client.version import MIN_API_VERSION
from aleph_alpha_client.aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionResult,
)
from aleph_alpha_client.prompt import Prompt
from tests.common import model_name, sync_client, async_client


def test_api_version_mismatch_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data("0.0.0")

    with pytest.raises(RuntimeError):
        Client(host=httpserver.url_for(""), token="AA_TOKEN").validate_version()


async def test_api_version_mismatch_async_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data("0.0.0")

    with pytest.raises(RuntimeError):
        async with AsyncClient(host=httpserver.url_for(""), token="AA_TOKEN") as client:
            await client.validate_version()


def test_api_version_correct_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data(MIN_API_VERSION)
    Client(host=httpserver.url_for(""), token="AA_TOKEN").validate_version()


async def test_api_version_correct_async_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data(MIN_API_VERSION)
    async with AsyncClient(host=httpserver.url_for(""), token="AA_TOKEN") as client:
        await client.validate_version()


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
            [
                CompletionResult(
                    log_probs=[],
                    completion="foo",
                )
            ],
            num_tokens_prompt_total=2,
            num_tokens_generated=1,
        ).to_json()
    )

    client = Client(host=httpserver.url_for(""), token="AA_TOKEN", nice=True)

    request = CompletionRequest(prompt=Prompt.from_text("Hello world"))
    client.complete(request, model="luminous")


async def test_nice_flag_on_async_client(httpserver: HTTPServer):
    httpserver.expect_request("/version").respond_with_data("OK")

    httpserver.expect_request(
        "/complete",
        query_string={"nice": "true"},
    ).respond_with_json(
        CompletionResponse(
            "model_version",
            [CompletionResult(log_probs=[], completion="foo")],
            num_tokens_prompt_total=2,
            num_tokens_generated=1,
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

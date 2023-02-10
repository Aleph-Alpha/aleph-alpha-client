from http import HTTPStatus
from random import choice
import re
import time
from typing import Optional
from aleph_alpha_client.aleph_alpha_client import (
    RETRY_STATUS_CODES,
    AlephAlphaClient,
    AsyncClient,
    BusyError,
    Client,
    _raise_for_status,
)
from aleph_alpha_client.aleph_alpha_model import AlephAlphaModel
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt
import pytest
import requests
from pytest_httpserver import HTTPServer
from requests.models import Response


def test_translate_errors():
    response = Response()
    response.code = "bad request"
    response.error_type = "bad request"
    response.status_code = 400
    response._content = b'{ "key" : "a" }'
    with pytest.raises(ValueError):
        _raise_for_status(response.status_code, response.text)


def test_retry_deprecated_sync(httpserver: HTTPServer):
    expect_retryable_error(httpserver, num_calls_expected=2)
    expect_valid_version(httpserver)

    # GETs /version
    AlephAlphaClient(host=httpserver.url_for(""), token="AA_TOKEN")


def test_retry_deprecated_sync_post(httpserver: HTTPServer):
    # required for initial GET /version in AlephAlphaClient init
    expect_valid_version(httpserver)
    client = AlephAlphaClient(host=httpserver.url_for(""), token="AA_TOKEN")
    model = AlephAlphaModel(client, "MODEL")
    expect_retryable_error(httpserver, num_calls_expected=2)
    expect_valid_completion(httpserver)

    request = CompletionRequest(prompt=Prompt.from_text(""), maximum_tokens=7)
    model.complete(request=request)


def test_retry_sync(httpserver: HTTPServer):
    num_retries = 2
    client = Client(
        token="AA_TOKEN", host=httpserver.url_for(""), total_retries=num_retries
    )
    expect_retryable_error(httpserver, num_calls_expected=num_retries)
    expect_valid_version(httpserver)

    client.get_version()


def test_retry_sync_post(httpserver: HTTPServer):
    num_retries = 2
    client = Client(
        host=httpserver.url_for(""), token="AA_TOKEN", total_retries=num_retries
    )
    expect_retryable_error(httpserver, num_calls_expected=num_retries)
    expect_valid_completion(httpserver)

    request = CompletionRequest(prompt=Prompt.from_text(""), maximum_tokens=7)
    client.complete(request=request, model="model")


def test_exhaust_retries_sync(httpserver: HTTPServer):
    num_retries = 1
    client = Client(
        token="AA_TOKEN", host=httpserver.url_for(""), total_retries=num_retries
    )
    expect_retryable_error(
        httpserver,
        num_calls_expected=num_retries + 1,
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    with pytest.raises(BusyError):
        client.get_version()


async def test_retry_async(httpserver: HTTPServer):
    num_retries = 2
    expect_retryable_error(httpserver, num_calls_expected=num_retries)
    expect_valid_version(httpserver)

    async with AsyncClient(
        token="AA_TOKEN", host=httpserver.url_for(""), total_retries=num_retries
    ) as client:
        await client.get_version()


async def test_retry_async_post(httpserver: HTTPServer):
    num_retries = 2
    expect_retryable_error(httpserver, num_calls_expected=num_retries)
    expect_valid_completion(httpserver)

    async with AsyncClient(
        token="AA_TOKEN", host=httpserver.url_for(""), total_retries=num_retries
    ) as client:
        request = CompletionRequest(prompt=Prompt.from_text(""), maximum_tokens=7)
        await client.complete(request, model="FOO")


async def test_exhaust_retries_async(httpserver: HTTPServer):
    num_retries = 1
    expect_retryable_error(
        httpserver,
        num_calls_expected=num_retries + 1,
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    with pytest.raises(BusyError):
        async with AsyncClient(
            token="AA_TOKEN", host=httpserver.url_for(""), total_retries=num_retries
        ) as client:
            await client.get_version()


# This test should stay last in this file because it usese a sleeping handler.
# This somehow blocks the plugin from clearing the httpserver state and thus
# causes crosstalk with other tests.
def test_timeout(httpserver: HTTPServer):
    def handler(foo):
        time.sleep(1)

    httpserver.expect_request("/version").respond_with_handler(handler)

    """Ensures Timeouts works. AlephAlphaClient constructor calls version endpoint."""
    with pytest.raises(requests.exceptions.ConnectionError):
        AlephAlphaClient(
            host=httpserver.url_for(""),
            token="AA_TOKEN",
            request_timeout_seconds=0.1,
            total_retries=1,
        )


def expect_retryable_error(
    httpserver: HTTPServer, num_calls_expected: int, status_code: Optional[int] = None
) -> None:
    for i in range(num_calls_expected):
        httpserver.expect_ordered_request(re.compile("^/")).respond_with_data(
            f"error({i})", status=status_code or choice(list(RETRY_STATUS_CODES))
        )


def expect_valid_completion(httpserver: HTTPServer) -> None:
    httpserver.expect_ordered_request("/complete").respond_with_json(
        {"model_version": "1", "completions": []}
    )


def expect_valid_version(httpserver: HTTPServer) -> None:
    httpserver.expect_ordered_request("/version").respond_with_data("ok", status=200)

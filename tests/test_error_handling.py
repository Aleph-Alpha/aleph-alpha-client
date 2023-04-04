from http import HTTPStatus
from random import choice
import re
from typing import Optional
from aleph_alpha_client.aleph_alpha_client import (
    RETRY_STATUS_CODES,
    AsyncClient,
    BusyError,
    Client,
    _raise_for_status,
)
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt
import pytest
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

from multiprocessing.sharedctypes import Value
import time
from http import HTTPStatus
from aleph_alpha_client.aleph_alpha_client import (
    AlephAlphaClient,
    AsyncClient,
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


def expect_call_respond_error(
    httpserver: HTTPServer, path: str, num_calls_expected: int, error_code: int
):
    for i in range(num_calls_expected):
        httpserver.expect_ordered_request(path).respond_with_data(
            f"error({i})", status=error_code
        )


def test_retry_sync_post(httpserver: HTTPServer):
    path = "/complete"
    httpserver.expect_ordered_request("/version").respond_with_data("busy1", status=200)
    expect_call_respond_error(
        httpserver,
        path,
        num_calls_expected=2,
        error_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    httpserver.expect_ordered_request(path).respond_with_json(
        {"model_version": "1", "completions": []}
    )

    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    client = AlephAlphaClient(host=httpserver.url_for(""), token="AA_TOKEN")
    model = AlephAlphaModel(client, "MODEL")
    model.complete(request=request)


def test_retry_sync(httpserver: HTTPServer):
    path = "/version"
    expect_call_respond_error(
        httpserver,
        path,
        num_calls_expected=2,
        error_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    httpserver.expect_ordered_request(path).respond_with_data("ok", status=200)

    AlephAlphaClient(host=httpserver.url_for(""), token="AA_TOKEN")


async def test_retry_async(httpserver: HTTPServer):
    path = "/version"
    expect_call_respond_error(
        httpserver,
        path,
        num_calls_expected=2,
        error_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    httpserver.expect_ordered_request(path).respond_with_data("ok", status=200)

    async with AsyncClient(token="AA_TOKEN", host=httpserver.url_for("")) as client:
        await client.get_version()


async def test_retry_async_post(httpserver: HTTPServer):
    path = "/complete"
    expect_call_respond_error(
        httpserver,
        path,
        num_calls_expected=2,
        error_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )
    httpserver.expect_ordered_request(path).respond_with_json(
        {"model_version": "1", "completions": []}
    )

    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    async with AsyncClient(token="AA_TOKEN", host=httpserver.url_for("")) as client:
        await client.complete(request, model="FOO")


# This test should stay last in this file because it usese a sleeping handler.
# This somehow blocks the plugin from clearing the httpserver state and thus
# causes crosstalk with other tests.
def test_timeout(httpserver: HTTPServer):
    def handler(foo):
        time.sleep(1)

    path = "/version"
    httpserver.expect_request(path).respond_with_handler(handler)

    """Ensures Timeouts works. AlephAlphaClient constructor calls version endpoint."""
    with pytest.raises(requests.exceptions.ConnectionError):
        AlephAlphaClient(
            host=httpserver.url_for(""), token="AA_TOKEN", request_timeout_seconds=0.1
        )

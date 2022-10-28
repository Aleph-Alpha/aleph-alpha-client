from multiprocessing.sharedctypes import Value
import time
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, BusyError
import pytest
import requests
from tests.common import client, model_name
from pytest_httpserver import HTTPServer
from requests.models import Response


def test_translate_errors():
    response = Response()
    response.code = "bad request"
    response.error_type = "bad request"
    response.status_code = 400
    response._content = b'{ "key" : "a" }'
    with pytest.raises(ValueError):
        AlephAlphaClient._translate_errors(response)


# setting a fixed port for httpserver
@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("127.0.0.1", 8000)


def test_timeout(httpserver: HTTPServer):
    def handler(foo):
        time.sleep(2)

    httpserver.expect_request("/version").respond_with_handler(handler)

    """Ensures Timeouts works. AlephAlphaClient constructor calls version endpoint."""
    with pytest.raises(requests.exceptions.ConnectionError):
        AlephAlphaClient(
            host="http://localhost:8000/", token="AA_TOKEN", request_timeout_seconds=0.1
        )


def test_retry_on_503(httpserver):
    httpserver.expect_request("/version").respond_with_data("busy", status=503)

    """Ensures Timeouts works. AlephAlphaClient constructor calls version endpoint."""
    with pytest.raises(BusyError):
        AlephAlphaClient(host="http://localhost:8000/", token="AA_TOKEN")


def test_retry_on_408(httpserver):
    httpserver.expect_request("/version").respond_with_data("timeout", status=408)

    """Ensures Timeouts works. AlephAlphaClient constructor calls version endpoint."""
    with pytest.raises(TimeoutError):
        AlephAlphaClient(host="http://localhost:8000/", token="AA_TOKEN")

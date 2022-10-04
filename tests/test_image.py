from http import HTTPStatus
from pytest import raises
from pytest_httpserver import HTTPServer
from requests import RequestException

from aleph_alpha_client.image import ImagePrompt


def test_from_url_with_non_OK_response(httpserver: HTTPServer):
    path = "/image"
    httpserver.expect_request(path).respond_with_data(
        "html", status=HTTPStatus.FORBIDDEN
    )

    with raises(RequestException) as e:
        ImagePrompt.from_url(httpserver.url_for(path))

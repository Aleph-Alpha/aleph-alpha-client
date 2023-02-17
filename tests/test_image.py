from http import HTTPStatus
from pathlib import Path

from pytest import raises
from pytest_httpserver import HTTPServer
from requests import RequestException

from aleph_alpha_client.image import Image, ImagePrompt


def test_from_url_with_non_OK_response(httpserver: HTTPServer):
    path = "/image"
    httpserver.expect_request(path).respond_with_data(
        "html", status=HTTPStatus.FORBIDDEN
    )

    with raises(RequestException) as e:
        Image.from_url(httpserver.url_for(path))


def test_from_image_source_on_local_file_path_pathlib():
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    img_1 = ImagePrompt.from_image_source(image_source=image_source_path)
    img_2 = ImagePrompt.from_file(path=image_source_path)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_local_file_path_str():
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    image_source_path = str(image_source_path)
    img_1 = ImagePrompt.from_image_source(image_source=image_source_path)
    img_2 = ImagePrompt.from_file(path=image_source_path)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_url():
    image_source_url = (
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    )
    img_1 = ImagePrompt.from_image_source(image_source=image_source_url)
    img_2 = ImagePrompt.from_url(url=image_source_url)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_bytes():
    with open(Path(__file__).parent / "dog-and-cat-cover.jpg", "rb") as f:
        image_source_bytes = f.read()

    img_1 = ImagePrompt.from_image_source(image_source=image_source_bytes)
    img_2 = ImagePrompt.from_bytes(bytes=image_source_bytes)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_bytes():
    with raises(TypeError):
        ImagePrompt.from_image_source(image_source=None)

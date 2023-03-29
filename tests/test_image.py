from http import HTTPStatus
from pathlib import Path

from pytest import raises
from pytest_httpserver import HTTPServer
from requests import RequestException

from aleph_alpha_client import Image


def test_from_url_with_non_OK_response(httpserver: HTTPServer):
    path = "/image"
    httpserver.expect_request(path).respond_with_data(
        "html", status=HTTPStatus.FORBIDDEN
    )

    with raises(RequestException) as e:
        Image.from_url(httpserver.url_for(path))


def test_from_image_source_on_local_file_path_pathlib():
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    str_image_source_path = str(image_source_path)

    img_1 = Image.from_image_source(image_source=image_source_path)
    img_2 = Image.from_file(path=str_image_source_path)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_local_file_path_str():
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    image_source_path = str(image_source_path)
    img_1 = Image.from_image_source(image_source=image_source_path)
    img_2 = Image.from_file(path=image_source_path)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_url():
    image_source_url = "https://docs.aleph-alpha.com/assets/images/room-8bc67118ba9576eaba51f284cb193394.jpg"
    img_1 = Image.from_image_source(image_source=image_source_url)
    img_2 = Image.from_url(url=image_source_url)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_bytes():
    image_source_bytes = (Path(__file__).parent / "dog-and-cat-cover.jpg").read_bytes()
    img_1 = Image.from_image_source(image_source=image_source_bytes)
    img_2 = Image.from_bytes(bytes=image_source_bytes)
    assert img_1.base_64 == img_2.base_64


def test_from_image_source_on_input_of_a_wrong_type():
    with raises(TypeError):
        Image.from_image_source(image_source=None)

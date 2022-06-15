import pytest
import requests
from tests.common import client


def test_health(client):
    response = requests.get(client.host + "health")
    assert (
        response.status_code == 200
    ), "requesting health endpoint returns status code 200"
    assert (
        response.text == "Ok"
    ), "requesting health endpoint does not return body 'Ok'; returned " + str(
        response.text
    )

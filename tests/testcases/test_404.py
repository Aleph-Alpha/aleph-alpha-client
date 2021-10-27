import pytest
import requests
from tests.common import client


def test_404(client):
    response = requests.get(client.host + "something_not_existing")
    assert response.status_code == 404, "requesting unknown endpoint returns status 404"
    data = response.json()
    assert (
        data["error"] == "not found"
    ), "requesting unknown endpoint returns json with property error"

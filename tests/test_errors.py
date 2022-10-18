import time
from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient, BusyError
import pytest
import requests
from tests.common import client, model_name
from pytest_httpserver import HTTPServer


@pytest.mark.parametrize(
    "config,description,exception_type",
    [
        ({}, "model must be provided", TypeError),
        ({"model": "test_model", "prompt": 123}, "prompt must be a string", ValueError),
        (
            {"model": "test_model", "maximum_tokens": "42"},
            "maximum_tokens must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "maximum_tokens": 1.2},
            "maximum_tokens must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "maximum_tokens": -1},
            "maximum_tokens must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "temperature": "2"},
            "temperature must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "temperature": -1},
            "temperature must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "temperature": 1.2},
            "temperature must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "top_k": "42"},
            "top_k must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "top_k": 1.2},
            "top_k must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "top_k": -1},
            "top_k must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "top_p": "2"},
            "top_p must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "top_p": -1},
            "top_p must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "top_p": 1.2},
            "top_p must be a float between 0 and 1",
            ValueError,
        ),
        (
            {"model": "test_model", "presence_penalty": "1.2"},
            "presence_penalty must be a number",
            ValueError,
        ),
        (
            {"model": "test_model", "frequency_penalty": "1.2"},
            "frequency_penalty must be a number",
            ValueError,
        ),
        (
            {"model": "test_model", "repetition_penalties_include_prompt": "1.2"},
            "repetition_penalties_include_prompt must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "repetition_penalties_include_prompt": 1},
            "repetition_penalties_include_prompt must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "repetition_penalties_include_prompt": 1.2},
            "repetition_penalties_include_prompt must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "use_multiplicative_presence_penalty": "1.2"},
            "use_multiplicative_presence_penalty must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "use_multiplicative_presence_penalty": 1},
            "use_multiplicative_presence_penalty must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "use_multiplicative_presence_penalty": 1.2},
            "use_multiplicative_presence_penalty must be a bool",
            ValueError,
        ),
        (
            {"model": "test_model", "best_of": 1, "n": 2},
            "best_of must be bigger than n",
            ValueError,
        ),
        (
            {"model": "test_model", "best_of": "abc"},
            "best_of must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "best_of": 1.5},
            "best_of must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "best_of": -1},
            "best_of must be a positive integer",
            ValueError,
        ),
        (
            {"model": "test_model", "n": "abc"},
            "n must be a positive integer",
            ValueError,
        ),
        ({"model": "test_model", "n": 1.5}, "n must be a positive integer", ValueError),
        ({"model": "test_model", "n": -1}, "n must be a positive integer", ValueError),
        (
            {"model": "test_model", "logit_bias": 2.2},
            "logit_bias must be a dict",
            ValueError,
        ),
        (
            {"model": "test_model", "logit_bias": "abc"},
            "logit_bias must be a dict",
            ValueError,
        ),
        (
            {"model": "test_model", "logit_bias": -1},
            "logit_bias must be a dict",
            ValueError,
        ),
        (
            {"model": "test_model", "logit_bias": list()},
            "logit_bias must be a dict",
            ValueError,
        ),
        (
            {"model": "test_model", "logit_bias": {"a": 1}},
            "logit_bias must be a dict mapping ints to a float",
            ValueError,
        ),
        (
            {"model": "test_model", "logit_bias": {1: "10"}},
            "logit_bias must be a dict mapping ints to a float",
            ValueError,
        ),
        (
            {"model": "test_model", "log_probs": "abc"},
            "n must be a positive integer or None",
            ValueError,
        ),
        (
            {"model": "test_model", "log_probs": 1.5},
            "n must be a positive integer or None",
            ValueError,
        ),
        (
            {"model": "test_model", "log_probs": -1},
            "n must be a positive integer or None",
            ValueError,
        ),
        (
            {"model": "test_model", "stop_sequences": 2.2},
            "stop_sequences must be a list of strings",
            ValueError,
        ),
        (
            {"model": "test_model", "stop_sequences": "abc"},
            "stop_sequences must be a list of strings",
            ValueError,
        ),
        (
            {"model": "test_model", "stop_sequences": -1},
            "stop_sequences must be a list of strings",
            ValueError,
        ),
        (
            {"model": "test_model", "stop_sequences": dict()},
            "stop_sequences must be a list of strings",
            ValueError,
        ),
        (
            {"model": "test_model", "stop_sequences": ["abc", 1]},
            "stop_sequences must be a list of strings",
            ValueError,
        ),
        ({"model": "test_model", "tokens": "1.2"}, "tokens must be a bool", ValueError),
        ({"model": "test_model", "tokens": 1}, "tokens must be a bool", ValueError),
        ({"model": "test_model", "tokens": 1.2}, "tokens must be a bool", ValueError),
        (
            {
                "model": "test_model",
                "prompt": " ".join(["abc" for _ in range(4000)]),
                "maximum_tokens": 1,
            },
            "too long prompt cannot be processed",
            ValueError,
        ),
    ],
)
@pytest.mark.needs_api
def test_aleph_alpha_client_completion_errors(
    client, model_name, config, description, exception_type
):
    if "model" in config:
        if config["model"] == "test_model":
            config["model"] = model_name

    with pytest.raises(exception_type):
        client.complete(**config)

    response = requests.post(
        client.host + "complete", headers=client.request_headers, json=config
    )
    assert response.status_code == 400, description


@pytest.mark.parametrize(
    "config,description,exception_type",
    [
        ({"completion_expected": "abc"}, "model must be provided", TypeError),
        (
            {"model": "test_model", "prompt": None, "completion_expected": "abc"},
            "prompt must be a string",
            ValueError,
        ),
        (
            {"model": "test_model", "prompt": 123, "completion_expected": "abc"},
            "prompt must be a string",
            ValueError,
        ),
        (
            {"model": "test_model", "prompt": "", "completion_expected": None},
            "completion_expected must be a string",
            ValueError,
        ),
        (
            {"model": "test_model", "prompt": "", "completion_expected": 123},
            "completion_expected must be a string",
            ValueError,
        ),
        (
            {"model": "test_model", "prompt": "", "completion_expected": ""},
            "completion_expected must have at least one character",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "prompt": " ".join(["abc" for _ in range(4000)]),
                "completion_expected": "abc",
            },
            "too long prompt cannot be processed",
            ValueError,
        ),
    ],
)
@pytest.mark.needs_api
def test_aleph_alpha_client_evaluation_errors(
    client, model_name, config, description, exception_type
):
    if "model" in config:
        if config["model"] == "test_model":
            config["model"] = model_name

    with pytest.raises(exception_type):
        client.evaluate(**config)

    response = requests.post(
        client.host + "evaluate", headers=client.request_headers, json=config
    )
    assert response.status_code == 400, (
        description
        + "; got status_code"
        + str(response.status_code)
        + " with body "
        + str(response.text)
    )


@pytest.mark.parametrize(
    "config,description,exception_type",
    [
        ({}, "model must be provided", TypeError),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "maximum_tokens": "42",
            },
            "maximum_tokens must be a positive integer",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "maximum_tokens": 1.2,
            },
            "maximum_tokens must be a positive integer",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "maximum_tokens": -1,
            },
            "maximum_tokens must be an int or None",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "max_chunk_size": "NOT AN INTEGER",
            },
            "max_chunk_size must be an int or None",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "max_answers": "NOT AN INTEGER",
            },
            "max_answers must be an int or None",
            ValueError,
        ),
        (
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "min_score": "NOT A FLOAT",
            },
            "min_score must be a float or None",
            ValueError,
        ),
    ],
)
@pytest.mark.needs_api
def test_aleph_alpha_client_qa_errors(
    client, model_name, config, description, exception_type
):
    if "model" in config:
        if config["model"] == "test_model":
            config["model"] = model_name

    with pytest.raises(exception_type):
        client.qa(**config)

    response = requests.post(
        client.host + "qa", headers=client.request_headers, json=config
    )
    assert response.status_code == 400, description


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

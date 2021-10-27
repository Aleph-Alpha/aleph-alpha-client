import pytest
import time
import requests
from aleph_alpha_client import QuotaError, POOLING_OPTIONS, ImagePrompt
from tests.common import client


def validate_completion_task_output(task, output):

    assert isinstance(output, dict), "completion result is a dict"
    assert "id" in output, "completion result has field id"
    assert "model_version" in output, "model_version in evaluation result"
    assert "completions" in output, "completion result has field completions"

    n = task.get("n", 1)
    assert n == len(output["completions"]), "got the right number of completions"
    for completion in output["completions"]:
        if task.get("tokens", False):
            assert "completion_tokens" in completion, "completion_tokens in completion"
        else:
            assert (
                "completion_tokens" not in completion
            ), "completion_tokens not in completion"

        if task.get("log_probs", -1) > -1:
            assert "log_probs" in completion, "log_probs in completion"
        else:
            assert "log_probs" not in completion, "log_probs not in completion"

        for field_name in ["completion", "finish_reason"]:
            assert field_name in completion, field_name + " in completion"

        assert isinstance(
            completion.get("completion", ""), str
        ), "completion is not a string"
        if task.get("tokens", False):
            assert (
                len(completion.get("completion_tokens", list()))
                == task["maximum_tokens"]
            ), (
                "generated "
                + str(len(completion.get("completion_tokens", list())))
                + " while "
                + str(task["maximum_tokens"])
                + " were requested"
            )
            if task.get("log_probs", -1) > -1:
                assert len(completion.get("log_probs", list())) == len(
                    completion.get("completion_tokens", dict())
                ), (
                    "generated "
                    + str(len(completion.get("log_probs", list())))
                    + " for "
                    + str(len(completion.get("completion_tokens", list())))
                    + " tokens"
                )

        for completion_token in completion.get("completion_tokens", list()):
            assert isinstance(completion_token, str), "completion_token is a str"

        for log_prob_dict in completion.get("log_probs", list()):
            assert isinstance(
                log_prob_dict, dict
            ), "token log prob is not a dict; got " + str(type(log_prob_dict))
            for token, log_prob in log_prob_dict.items():
                assert isinstance(
                    token, str
                ), "token in log prob is not a str; got " + str(type(token))
                assert isinstance(
                    log_prob, float
                ), "log_prob in log prob is not a float; got " + str(type(log_prob))


def validate_evaluation_task_output(task, output):
    assert isinstance(output, dict), "result is a dict, got " + str(type(output))

    for field_name in ["message"]:
        assert field_name in output, field_name + " in output"
        assert (
            isinstance(output.get(field_name), str) or output.get(field_name) is None
        ), (
            field_name
            + " is not a str or None; got "
            + str(type(output.get(field_name)))
        )

    assert "id" in output, "id in evaluation result"
    assert "model_version" in output, "model_version in evaluation result"
    assert "result" in output, "result dict in evaluation output"

    for field_name in [
        "log_probability",
        "log_perplexity",
        "log_perplexity_per_token",
        "log_perplexity_per_character",
    ]:
        assert field_name in output["result"], field_name + " not in result"
        assert isinstance(output["result"].get(field_name), float), (
            field_name
            + " is not a float; got "
            + str(type(output["result"].get(field_name)))
        )

    for field_name in ["correct_greedy"]:
        assert field_name in output["result"], field_name + " not in result"
        assert isinstance(output["result"].get(field_name), bool), (
            field_name
            + " is not a bool; got "
            + str(type(output["result"].get(field_name)))
        )

    for field_name in ["token_count", "character_count"]:
        assert field_name in output["result"], field_name + " not in result"
        assert isinstance(output["result"].get(field_name), int), (
            field_name
            + " is not an int; got "
            + str(type(output["result"].get(field_name)))
        )

    for field_name in ["completion"]:
        assert field_name in output["result"], field_name + " not in result"
        assert isinstance(output["result"].get(field_name), str), (
            field_name
            + " is not a str; got "
            + str(type(output["result"].get(field_name)))
        )


def validate_embedding_task_output(task, output):
    if not "pooling" in task:
        task["pooling"] = None

    assert isinstance(output, dict), "output is a dict, got " + str(type(output))

    for field_name in ["message"]:
        assert field_name in output, field_name + " in output"
        assert (
            isinstance(output.get(field_name), str) or output.get(field_name) is None
        ), (
            field_name
            + " is not a str or None; got "
            + str(type(output.get(field_name)))
        )

    assert "id" in output, "id in evaluation result"
    assert "model_version" in output, "model_version in evaluation result"

    assert "embeddings" in output, "output contains embeddings"
    assert isinstance(output["embeddings"], dict), "embeddings is a dict"
    assert len(output["embeddings"]) == len(
        task["layers"]
    ), "embeddings contain one embedding per layer"

    if task["pooling"] is None or len(task["pooling"]) == 0:
        for layer_embeddings in output["embeddings"].values():
            assert isinstance(
                layer_embeddings, list
            ), "no pooling, layer embeddings are lists"
            for token_embedding in layer_embeddings:
                assert isinstance(
                    token_embedding, list
                ), "no pooling, token embeddings are a list"
                for value in token_embedding:
                    assert isinstance(
                        value, float
                    ), "no pooling, a value of a token embedding is a float"
    else:
        for layer_embeddings in output["embeddings"].values():
            assert isinstance(
                layer_embeddings, dict
            ), "pooling, layer embeddings are dicts"
            assert len(layer_embeddings) == len(
                task["pooling"]
            ), "embeddings contain one embedding per layer"
            for pooling_option, pooled_embedding in layer_embeddings.items():
                assert pooling_option in POOLING_OPTIONS, (
                    "pooling option is in "
                    + str(POOLING_OPTIONS)
                    + ", got "
                    + str(pooling_option)
                )
                for value in pooled_embedding:
                    assert isinstance(
                        value, float
                    ), "pooling, a value of a pooled embedding is a float"


@pytest.mark.parametrize(
    "endpoint,task_definition",
    [
        (
            "complete",
            {
                "model": "test_model",
                "prompt": "",
                "maximum_tokens": 7,
                "tokens": False,
                "log_probs": 0,
            },
        ),
        (
            "complete",
            {
                "model": "test_model",
                "prompt": "",
                "maximum_tokens": 7,
                "tokens": True,
                "log_probs": 0,
            },
        ),
        (
            "complete",
            {
                "model": "test_model",
                "prompt": "",
                "maximum_tokens": 7,
                "tokens": True,
                "log_probs": 2,
            },
        ),
        (
            "evaluate",
            {"model": "test_model", "prompt": "", "completion_expected": "abc"},
        ),
        (
            "embed",
            {
                "model": "test_model",
                "prompt": "abc",
                "layers": [-1],
                "pooling": ["mean"],
            },
        ),
        (
            "embed",
            {
                "model": "test_model",
                "prompt": "abc",
                "layers": [-1],
                "pooling": ["mean", "max"],
            },
        ),
    ],
)
def test_task(client, endpoint, task_definition):
    if "model" in task_definition:
        if task_definition["model"] == "test_model":
            task_definition["model"] = client.test_model

    # start a task in a thread in order to run worker in between
    if endpoint == "complete":
        result = client.complete(**task_definition)
    elif endpoint == "embed":
        result = client.embed(**task_definition)
    elif endpoint == "evaluate":
        result = client.evaluate(**task_definition)

    if endpoint == "complete":
        validate_completion_task_output(task_definition, result)
    elif endpoint == "evaluate":
        validate_evaluation_task_output(task_definition, result)
    elif endpoint == "embed":
        validate_embedding_task_output(task_definition, result)


def test_should_answer_question_about_image(client):

    # Only execute this test if the model has multimodal support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == client.test_model, models))
    if not model["image_support"]:
        return

    prompt = [
        "Q: What shop can be seen in the picture? A:",
        ImagePrompt.from_file("./tests/image_example.jpg"),
    ]

    result = client.complete(
        model=client.test_model, prompt=prompt, maximum_tokens=64, tokens=False
    )
    print(result)

    assert "Blockbuster" in result["completions"][0]["completion"]

import pytest
import time
from aleph_alpha_client import QuotaError, POOLING_OPTIONS, Image, Document
from tests.common import client, model_name


@pytest.mark.system_test
def validate_completion_task_output(task, output):

    assert isinstance(output, dict), "completion result is a dict"
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


@pytest.mark.system_test
def validate_evaluation_task_output(task, output):
    assert isinstance(output, dict), "result is a dict, got " + str(type(output))

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


@pytest.mark.system_test
def validate_embedding_task_output(task, output):
    if not "pooling" in task:
        task["pooling"] = None

    assert isinstance(output, dict), "output is a dict, got " + str(type(output))

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


@pytest.mark.system_test
def validate_qa_task_output(task, output):

    assert isinstance(output, dict), "qa result is a dict"
    assert "model_version" in output, "model_version in qa result"
    assert "answers" in output, "qa result has field answers"

    for answer in output["answers"]:
        for field_name in [
            "answer",
            "score",
            "evidence",
        ]:
            assert field_name in answer, field_name + " field is missing in answer"


@pytest.mark.parametrize(
    "model_name,endpoint,task_definition",
    [
        (
            "luminous-base",
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
            "luminous-base",
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
            "luminous-base",
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
            "luminous-base",
            "evaluate",
            {"model": "test_model", "prompt": "", "completion_expected": "abc"},
        ),
        (
            "luminous-extended",
            "qa",
            {
                "model": "test_model",
                "query": "",
                "documents": [],
                "maximum_tokens": 7,
            },
        ),
        (
            "luminous-extended",
            "qa",
            {
                "model": "test_model",
                "query": "Who likes to eat pizza?",
                "documents": [Document.from_docx_file("tests/sample.docx")],
                "maximum_tokens": 64,
            },
        ),
    ],
)
@pytest.mark.system_test
def test_task(client, model_name, endpoint, task_definition):
    if "model" in task_definition:
        if task_definition["model"] == "test_model":
            task_definition["model"] = model_name

    # start a task in a thread in order to run worker in between
    if endpoint == "complete":
        result = client.complete(**task_definition)
    elif endpoint == "embed":
        result = client.embed(**task_definition)
    elif endpoint == "evaluate":
        result = client.evaluate(**task_definition)
    elif endpoint == "qa":
        result = client.qa(**task_definition)

    if endpoint == "complete":
        validate_completion_task_output(task_definition, result)
    elif endpoint == "evaluate":
        validate_evaluation_task_output(task_definition, result)
    elif endpoint == "embed":
        validate_embedding_task_output(task_definition, result)
    elif endpoint == "qa":
        validate_qa_task_output(task_definition, result)


@pytest.mark.system_test
def test_should_answer_question_about_image(client, model_name):

    # Only execute this test if the model has multimodal support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == model_name, models))
    if not model["image_support"]:
        return

    prompt = [
        "Q: What shop can be seen in the picture? A:",
        Image.from_file("./tests/image_example.jpg"),
    ]

    result = client.complete(
        model=model_name, prompt=prompt, maximum_tokens=64, tokens=False
    )
    print(result)

    assert "Blockbuster" in result["completions"][0]["completion"]


@pytest.mark.system_test
def test_should_entertain_image_cropping_params(client, model_name):
    # Only execute this test if the model has multimodal support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == model_name, models))
    if not model["image_support"]:
        return

    prompt = [
        Image.from_file_with_cropping("./tests/dog-and-cat-cover.jpg", 0, 0, 630),
    ]

    result = client.complete(
        model=model_name, prompt=prompt, maximum_tokens=64, tokens=False
    )
    print(result)

    assert "dog" in result["completions"][0]["completion"].lower()


@pytest.mark.system_test
def test_should_answer_query_about_docx_document(client, model_name):

    # Only execute this test if the model has qa support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == model_name, models))
    if not model["qa_support"]:
        return

    query = "Who likes to eat pizza?"

    document = Document.from_docx_file(
        "tests/sample.docx"
    )  # With content: "Markus likes to eat pizza.\n\nBen likes to eat Pasta.\n\nDenis likes to eat Hotdogs."
    documents = [document]

    result = client.qa(
        model=model_name, query=query, documents=documents, maximum_tokens=64
    )

    assert "Markus" in result["answers"][0]["answer"]


@pytest.mark.system_test
def test_should_answer_query_about_docx_bytes_document(client, model_name):

    # Only execute this test if the model has qa & multimodal support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == model_name, models))
    if not model["qa_support"]:
        return

    query = "Who likes to eat pizza?"

    # Turn sample docx into bytes to test `from_docx_bytes` functionality
    sample_file = "tests/sample.docx"
    with open("tests/sample.docx", "rb") as sample_file:
        sample_data = sample_file.read()
        document = Document.from_docx_bytes(
            sample_data
        )  # With content: "Markus likes to eat pizza.\n\nBen likes to eat Pasta.\n\nDenis likes to eat Hotdogs."

        documents = [document]

        result = client.qa(
            model=model_name, query=query, documents=documents, maximum_tokens=64
        )

        assert "Markus" in result["answers"][0]["answer"]


@pytest.mark.system_test
def test_should_answer_query_about_prompt_document(client, model_name):

    # Only execute this test if the model has qa support
    models = client.available_models()
    model = next(filter(lambda model: model["name"] == model_name, models))
    if not (model["qa_support"] and model["image_support"]):
        return

    query = "Who likes to eat pizza?"

    prompt = ["Markus likes to eat pizza."]
    document = Document.from_prompt(prompt)
    documents = [document]

    result = client.qa(
        model=model_name, query=query, documents=documents, maximum_tokens=64
    )

    assert "Markus" in result["answers"][0]["answer"]

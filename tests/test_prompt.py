from aleph_alpha_client import Prompt, Tokens, TokenControl


def test_serialize_token_ids():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(Tokens(tokens))
    serialize = prompt.to_json()

    prompt_item = serialize[0]

    assert prompt_item["type"] == "token_ids"
    assert prompt_item["data"] == tokens


def test_serialize_token_ids_with_controls():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(
        Tokens(
            tokens,
            controls=[
                TokenControl(index=0, factor=0.25),
                TokenControl(index=1, factor=0.5),
            ],
        )
    )
    serialize = prompt.to_json()

    prompt_item = serialize[0]

    assert prompt_item["type"] == "token_ids"
    assert prompt_item["data"] == tokens
    assert prompt_item["controls"][0]["index"] == 0
    assert prompt_item["controls"][0]["factor"] == 0.25
    assert prompt_item["controls"][1]["index"] == 1
    assert prompt_item["controls"][1]["factor"] == 0.5

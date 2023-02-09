from aleph_alpha_client import Prompt, Tokens


def test_serialize_token_ids():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(Tokens(tokens))
    serialize = prompt._serialize()

    prompt_item = serialize[0]

    assert prompt_item["type"] == "token_ids"
    assert prompt_item["data"] == tokens

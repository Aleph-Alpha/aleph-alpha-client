from aleph_alpha_client import Prompt, Tokens, TokenControl


def test_serialize_token_ids():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(Tokens(tokens))
    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [{"type": "token_ids", "data": [1, 2, 3, 4]}]


def test_serialize_token_ids_with_controls():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(
        Tokens(
            tokens,
            controls=[
                TokenControl(pos=0, factor=0.25),
                TokenControl(pos=1, factor=0.5),
            ],
        )
    )
    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [
        {
            "type": "token_ids",
            "data": tokens,
            "controls": [{"index": 0, "factor": 0.25}, {"index": 1, "factor": 0.5}],
        }
    ]

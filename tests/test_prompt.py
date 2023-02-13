from aleph_alpha_client import Prompt, Tokens, TokenControl
from aleph_alpha_client.prompt import TextControl


def test_serialize_token_ids():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(tokens)
    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [
        {"type": "token_ids", "data": [1, 2, 3, 4], "controls": []}
    ]


def test_serialize_token_ids_with_controls():
    tokens = [1, 2, 3, 4]
    prompt = Prompt.from_tokens(
        tokens,
        controls=[
            TokenControl(pos=0, factor=0.25),
            TokenControl(pos=1, factor=0.5),
        ],
    )
    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [
        {
            "type": "token_ids",
            "data": tokens,
            "controls": [{"index": 0, "factor": 0.25}, {"index": 1, "factor": 0.5}],
        }
    ]


def test_serialize_text_with_controls():
    prompt_text = "An apple a day"
    prompt = Prompt.from_text(prompt_text, [TextControl(start=3, length=5, factor=1.5)])

    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [
        {
            "type": "text",
            "data": prompt_text,
            "controls": [{"start": 3, "length": 5, "factor": 1.5}],
        }
    ]

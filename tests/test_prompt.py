from aleph_alpha_client import Prompt, Tokens, TokenControl, Image, ImageControl
from aleph_alpha_client.aleph_alpha_client import Client
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import Text, TextControl
from tests.common import sync_client, model_name


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


def test_serialize_image_with_controls():
    image = Image.from_file(
        "tests/dog-and-cat-cover.jpg", [ImageControl(0.0, 0.0, 0.5, 0.5, 0.5)]
    )
    prompt = Prompt.from_image(image)
    serialized_prompt = prompt.to_json()

    assert serialized_prompt == [
        {
            "type": "image",
            "data": image.base_64,
            "controls": [
                {
                    "rect": {
                        "left": 0.0,
                        "top": 0.0,
                        "width": 0.5,
                        "height": 0.5,
                    },
                    "factor": 0.5,
                }
            ],
        }
    ]


def test_image_controls_with_cats_and_dogs(sync_client: Client, model_name: str):
    image = ImagePrompt.from_file_with_cropping(
        "tests/dog-and-cat-cover.jpg",
        # crop exactly 600x600 pixels out of the image
        300,
        0,
        600,
        controls=[
            # Suppress the cat
            ImageControl(left=0.5, top=0.0, width=0.25, height=1.0, factor=0.0)
        ],
    )
    text = Text.from_text("A picture of ")
    request = CompletionRequest(
        prompt=Prompt([image, text]),
        maximum_tokens=16,
        control_log_additive=True,
        disable_optimizations=False,
    )
    result = sync_client.complete(request, model="luminous-extended")
    assert result.completions[0].completion == " a dog"

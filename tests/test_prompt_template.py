from pathlib import Path
from pytest import raises
from aleph_alpha_client.prompt import Prompt, Image, Text, Tokens
from aleph_alpha_client.prompt_template import PromptTemplate
from liquid.exceptions import LiquidTypeError
from .common import prompt_image


def test_to_prompt_with_text_array():
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )
    names = ["World", "Rutger"]

    prompt = template.to_prompt(names=names)

    expected = "".join([f"Hello {name}!\n" for name in names])
    assert prompt == Prompt.from_text(expected)


def test_to_prompt_with_invalid_input():
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )

    with raises(LiquidTypeError):
        template.to_prompt(names=7)


def test_to_prompt_with_single_image(prompt_image: Image):
    template = PromptTemplate(
        """Some Text.
{{whatever}}
More Text
"""
    )

    prompt = template.to_prompt(whatever=template.placeholder(prompt_image))

    expected = Prompt(
        [
            Text.from_text("Some Text.\n"),
            prompt_image,
            Text.from_text("\nMore Text\n"),
        ]
    )
    assert prompt == expected


def test_to_prompt_with_image_sequence(prompt_image: Image):
    template = PromptTemplate(
        """
{%- for image in images -%}
{{image}}
{%- endfor -%}
        """
    )

    prompt = template.to_prompt(
        images=[template.placeholder(prompt_image), template.placeholder(prompt_image)]
    )

    expected = Prompt([prompt_image, prompt_image])
    assert prompt == expected


def test_to_prompt_with_mixed_modality_variables(prompt_image: Image):
    template = PromptTemplate("""{{image}}{{name}}{{image}}""")

    prompt = template.to_prompt(
        image=template.placeholder(prompt_image), name="whatever"
    )

    expected = Prompt([prompt_image, Text.from_text("whatever"), prompt_image])
    assert prompt == expected


def test_to_prompt_with_unused_image(prompt_image: Image):
    template = PromptTemplate("cool")

    prompt = template.to_prompt(images=template.placeholder(prompt_image))

    assert prompt == Prompt.from_text("cool")


def test_to_prompt_with_multiple_different_images(prompt_image: Image):
    image_source_path = Path(__file__).parent / "image_example.jpg"
    second_image = Image.from_file(image_source_path)

    template = PromptTemplate("""{{image_1}}{{image_2}}""")

    prompt = template.to_prompt(
        image_1=template.placeholder(prompt_image),
        image_2=template.placeholder(second_image),
    )

    assert prompt == Prompt([prompt_image, second_image])


def test_to_prompt_with_embedded_prompt(prompt_image: Image):
    user_prompt = Prompt([Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("""{{user_prompt}}""")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_does_not_add_whitespace_after_image(prompt_image: Image):
    user_prompt = Prompt([prompt_image, Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_skips_empty_strings():
    user_prompt = Prompt(
        [Text.from_text("Cool"), Text.from_text(""), Text.from_text("Also cool")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == Prompt([Text.from_text("Cool Also cool")])


def test_to_prompt_adds_whitespaces():
    user_prompt = Prompt(
        [Text.from_text("start "), Text.from_text("middle"), Text.from_text(" end")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == Prompt([Text.from_text("start middle end")])


def test_to_prompt_works_with_tokens():
    user_prompt = Prompt(
        [
            Tokens.from_token_ids([1, 2, 3]),
            Text.from_text("cool"),
            Tokens.from_token_ids([4, 5, 6]),
        ]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt

def test_to_prompt_resets_cache(prompt_image: Image):
    user_prompt = Prompt([prompt_image, Text.from_text("Cool"), prompt_image])
    
    template = PromptTemplate("{{user_prompt}}")

    template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert template.non_text_items == {} 

from pathlib import Path
from pytest import raises
from aleph_alpha_client.prompt import Prompt, Image, Text
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

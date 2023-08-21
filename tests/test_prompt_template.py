from pathlib import Path
from pytest import raises
from aleph_alpha_client.prompt import Prompt, Image, Text
from aleph_alpha_client.prompt_template import PromptTemplate
from liquid.exceptions import LiquidTypeError


def test_to_prompt_with_array():
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

def test_to_prompt_with_image():
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    image = Image.from_file(image_source_path)
    template = PromptTemplate(
        """Some Text.
{{whatever}}
More Text
"""
    )

    prompt = template.to_prompt(whatever=image)

    expected = Prompt([
        Text.from_text("Some Text.\n"),
        image,
        Text.from_text("More Text\n"),
    ]) 
    assert prompt == expected 



from aleph_alpha_client.prompt import Prompt
from aleph_alpha_client.prompt_template import PromptTemplate


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

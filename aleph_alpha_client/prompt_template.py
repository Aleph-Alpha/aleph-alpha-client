import uuid
from liquid import Template

from aleph_alpha_client.prompt import Image, Prompt


class PromptTemplate:
    """Allows to build a `Prompt` using the `liquid template language <https://shopify.github.io/liquid/>`_.

    Example:
        >>> template = PromptTemplate(
            '''{%- for name in names -%}
            Hello {{name}}!
            {% endfor -%}
            ''')
        >>> names = ["World", "Rutger"]
        >>> prompt = template.to_prompt(names=names)
        >>> request = CompletionRequest(prompt=prompt)
    """

    def __init__(self, template_str: str) -> None:
        """Initialize with the liquid template string.

        Parameters:
            template_str: the liquid template string
        """
        self.template = Template(template_str)

    def to_prompt(self, **kwargs) -> Prompt:
        """Creates a `Prompt` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        images = []
        placeholders = []
        for _, value in kwargs.items():
            if isinstance(value, Image):
                images.append(value)
                value = uuid()
                placeholders.append(value)

        liquid_prompt: str = self.template.render(**kwargs)

        split_prompt = []
        for placeholder in placeholders:
            # split string on regex
            split_prompt = liquid_prompt.(placeholder, 1)

        # turn split_prompt into AA prompt 

        Prompt.from_text(self.template.render(**kwargs))

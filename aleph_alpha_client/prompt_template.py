from re import finditer
from typing import Iterable, Mapping, Sequence, Tuple
from uuid import UUID, uuid4
from liquid import Template

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text


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
        image_by_placeholder = {}
        updated_args = dict(**kwargs)
        for arg, value in kwargs.items():
            if isinstance(value, Image):
                placeholder = str(uuid4())
                image_by_placeholder[placeholder] = value
                updated_args[arg] = placeholder

        liquid_prompt: str = self.template.render(**updated_args)

        placeholder_indices = compute_indices(
            image_by_placeholder.keys(), liquid_prompt
        )
        modalities = modalities_from(
            placeholder_indices, image_by_placeholder, liquid_prompt
        )

        return Prompt(list(modalities))


def compute_indices(
    placeholders: Iterable[str], template: str
) -> Iterable[Tuple[int, int]]:
    pattern = f"({'|'.join(placeholders)})"
    return ((match.start(), match.end()) for match in finditer(pattern, template))


def modalities_from(
    placeholder_indices: Iterable[Tuple[int, int]],
    image_by_placeholder: Mapping[str, Image],
    template: str,
) -> Iterable[PromptItem]:
    last_to = 0
    for placeholder_from, placeholder_to in placeholder_indices:
        if last_to < placeholder_from:
            yield Text.from_text(template[last_to:placeholder_from])
        yield image_by_placeholder[template[placeholder_from:placeholder_to]]
        last_to = placeholder_to + 1
    if last_to < len(template):
        yield Text.from_text(template[last_to:])

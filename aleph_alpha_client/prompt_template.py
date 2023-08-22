from re import finditer
from typing import Dict, Iterable, Mapping, NewType, Tuple
from uuid import UUID, uuid4
from liquid import Template

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text

Placeholder = NewType("Placeholder", UUID)


class PromptTemplate:
    """Allows to build a `Prompt` using the `liquid template language <https://shopify.github.io/liquid/>`_.

    To add images to the prompt first you have to save it to the template with the `template.placeholder()` function.
    To embed the image in the template, pass the placeholder in the place(s) where you would like the image.

    Example:
        >>> image = Image.from_file(Path("path-to-image"))
        >>> template = PromptTemplate(
            '''{%- for name in names -%}
            Hello {{name}}!
            {% endfor -%}
            {{ image }}
            ''')
        >>> placeholder = template.placeholder(image)
        >>> names = ["World", "Rutger"]
        >>> prompt = template.to_prompt(names=names, image=placeholder)
        >>> request = CompletionRequest(prompt=prompt)
    """

    def __init__(self, template_str: str) -> None:
        """Initialize with the liquid template string.

        Parameters:
            template_str: the liquid template string
        """
        self.template = Template(template_str)
        self.images: Dict[Placeholder, Image] = {}

    def placeholder(self, image: Image) -> Placeholder:
        """Saves an image to the template and returns a placeholder

        The placeholder is used to embed the image in the template
        """
        id = Placeholder(uuid4())
        self.images[id] = image
        return id

    def to_prompt(self, **kwargs) -> Prompt:
        """Creates a `Prompt` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        liquid_prompt: str = self.template.render(**kwargs)
        placeholder_indices = self._compute_indices(self.images.keys(), liquid_prompt)
        modalities = _modalities_from(placeholder_indices, self.images, liquid_prompt)

        return Prompt(list(modalities))

    def _compute_indices(
        self, placeholders: Iterable[Placeholder], template: str
    ) -> Iterable[Tuple[int, int]]:
        if not self.images:
            return []
        pattern = f"({'|'.join(str(placeholder) for placeholder in placeholders)})"
        return ((match.start(), match.end()) for match in finditer(pattern, template))


def _modalities_from(
    placeholder_indices: Iterable[Tuple[int, int]],
    image_by_placeholder: Mapping[Placeholder, Image],
    template: str,
) -> Iterable[PromptItem]:
    last_to = 0
    for placeholder_from, placeholder_to in placeholder_indices:
        if last_to < placeholder_from:
            yield Text.from_text(template[last_to:placeholder_from])
        yield image_by_placeholder[
            Placeholder(UUID(template[placeholder_from:placeholder_to]))
        ]
        last_to = placeholder_to
    if last_to < len(template):
        yield Text.from_text(template[last_to:])

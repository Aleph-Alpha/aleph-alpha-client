from re import finditer
from typing import Dict, Iterable, Mapping, NewType, Tuple, Union
from uuid import UUID, uuid4
from liquid import Template

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text, Tokens

Placeholder = NewType("Placeholder", UUID)


class PromptTemplate:
    """Allows to build a `Prompt` using the `liquid template language <https://shopify.github.io/liquid/>`_.

    To add non-text prompt items first you have to save it to the template with the `template.placeholder()` function.
    To embed the items in the template, pass the placeholder in the place(s) where you would like the items.

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
        self.non_text_items: Dict[Placeholder, Union[Image, Tokens]] = {}

    def placeholder(self, prompt_item: Union[Image, Tokens]) -> Placeholder:
        """Saves a non-text prompt item to the template and returns a placeholder

        The placeholder is used to embed the prompt item in the template
        """
        id = Placeholder(uuid4())
        self.non_text_items[id] = prompt_item
        return id

    def _join_character(
        self, first_item: Union[Text, Image, Tokens, None], second_item: Text
    ) -> str:
        if (
            isinstance(first_item, Text)
            and not first_item.text[-1].isspace()
            and not second_item.text[0].isspace()
        ):
            return " "
        else:
            return ""

    def embed_prompt(self, prompt: Prompt) -> str:
        """Embeds a prompt in a prompt template

        Adds whitespace between text items if there is no whitespace between them.
        In case of non-text prompt items, this embeds them into the end result.

        Example:
            >>> user_prompt = Prompt(
                    [
                        Tokens.from_token_ids([1, 2, 3]),
                        Text.from_text("cool"),
                        Image.from_file(Path("path-to-image")),
                    ]
                )
            >>> template = PromptTemplate("Question: {{user_prompt}}\\n Answer: ")
            >>> prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

        Parameters:
            prompt: prompt to embed in the template
        """
        prompt_text = ""
        last_item = None
        for item in prompt.items:
            if isinstance(item, Text):
                if len(item.text) == 0:
                    continue
                prompt_text = str.join(
                    self._join_character(last_item, item), [prompt_text, item.text]
                )
            else:
                prompt_text = str.join("", [prompt_text, str(self.placeholder(item))])
            last_item = item
        return prompt_text

    def to_prompt(self, **kwargs) -> Prompt:
        """Creates a `Prompt` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        liquid_prompt: str = self.template.render(**kwargs)
        placeholder_indices = self._compute_indices(
            self.non_text_items.keys(), liquid_prompt
        )
        modalities = _modalities_from(
            placeholder_indices, self.non_text_items, liquid_prompt
        )

        self.non_text_items = {}
        return Prompt(list(modalities))

    def _compute_indices(
        self, placeholders: Iterable[Placeholder], template: str
    ) -> Iterable[Tuple[int, int]]:
        if not self.non_text_items:
            return []
        pattern = f"({'|'.join(str(placeholder) for placeholder in placeholders)})"
        return ((match.start(), match.end()) for match in finditer(pattern, template))


def _modalities_from(
    placeholder_indices: Iterable[Tuple[int, int]],
    prompt_items_by_placeholder: Mapping[Placeholder, Union[Image, Tokens]],
    template: str,
) -> Iterable[PromptItem]:
    last_to = 0
    for placeholder_from, placeholder_to in placeholder_indices:
        if last_to < placeholder_from:
            yield Text.from_text(template[last_to:placeholder_from])
        yield prompt_items_by_placeholder[
            Placeholder(UUID(template[placeholder_from:placeholder_to]))
        ]
        last_to = placeholder_to
    if last_to < len(template):
        yield Text.from_text(template[last_to:])

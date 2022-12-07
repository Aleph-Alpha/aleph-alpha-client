from typing import Any, Dict, List, NamedTuple, Union

from aleph_alpha_client.image import ImagePrompt


class Prompt(NamedTuple):
    """
    Examples:
        >>> prompt = Prompt.from_text("Provide a short description of AI:")
        >>> prompt = Prompt([
                ImagePrompt.from_url(url),
                "Provide a short description of AI:",
            ])
    """

    items: List[Union[str, ImagePrompt, List[int]]]

    @staticmethod
    def from_text(text: str) -> "Prompt":
        return Prompt([text])

    @staticmethod
    def from_image(image: ImagePrompt) -> "Prompt":
        return Prompt([image])

    @staticmethod
    def from_tokens(tokens: List[int]) -> "Prompt":
        return Prompt([tokens])


def _to_prompt_item(item: Union[str, ImagePrompt, List[int]]) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"type": "text", "data": item}
    elif isinstance(item, List):
        return {"type": "token_ids", "data": item}
    elif hasattr(item, "_to_prompt_item"):
        return item._to_prompt_item()
    else:
        raise ValueError(
            "The item in the prompt is not valid. Try either a string or an Image."
        )


def _to_serializable_prompt(
    prompt, at_least_one_token=False
) -> Union[str, List[Dict[str, str]]]:
    """
    Validates that a prompt and emits the format suitable for serialization as JSON
    """
    if isinstance(prompt, str):
        if at_least_one_token:
            if len(prompt) == 0:
                raise ValueError("prompt must contain at least one character")
        # Just pass the string through as is.
        return prompt

    elif isinstance(prompt, list):
        return [_to_prompt_item(item) for item in prompt]

    raise ValueError(
        "Invalid prompt. Prompt must either be a string, or a list of valid multimodal propmt items."
    )

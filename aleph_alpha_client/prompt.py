from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

from aleph_alpha_client.image import Image


class TokenControl(NamedTuple):
    """
    Used for Attention Manipulation, for a given token index, you can supply
    the factor you want to adjust the attention by.

    Parameters:
        pos (int, required):
            The index of the token in the prompt item that you want to apply
            the factor to.

        factor (float, required):
            The amount to adjust model attention by.
            Values between 0 and 1 will supress attention.
            A value of 1 will have no effect.
            Values above 1 will increase attention.

    Examples:
        >>> Tokens([1, 2, 3], controls=[TokenControl(pos=1, factor=0.5)])
    """

    pos: int
    factor: float

    def to_json(self) -> Mapping[str, Any]:
        return {"index": self.pos, "factor": self.factor}


class Tokens(NamedTuple):
    """
    A list of token ids to be sent as part of a prompt.

    Parameters:
        tokens (List(int), required):
            The tokens you want to be passed to the model as part of your prompt.

        controls (List(TokenControl), optional, default None):
            DISCLAIMER: This may not be supported at the time of package release.

            Used for Attention Manipulation. Provides the ability to change
            attention for given token ids.

    Examples:
        >>> token_ids = Tokens([1, 2, 3])
        >>> prompt = Prompt([token_ids])
    """

    tokens: Sequence[int]
    controls: Optional[Sequence[TokenControl]] = None

    def to_json(self) -> Mapping[str, Any]:
        """
        Serialize the prompt item to JSON for sending to the API.
        """
        payload = {"type": "token_ids", "data": self.tokens}
        if self.controls:
            payload["controls"] = [c.to_json() for c in self.controls]
        return payload


class Prompt(NamedTuple):
    """
    Examples:
        >>> prompt = Prompt.from_text("Provide a short description of AI:")
        >>> prompt = Prompt([
                Image.from_url(url),
                "Provide a short description of AI:",
            ])
    """

    items: Sequence[Union[str, Image, Tokens, Sequence[int]]]

    @staticmethod
    def from_text(text: str) -> "Prompt":
        return Prompt([text])

    @staticmethod
    def from_image(image: Image) -> "Prompt":
        return Prompt([image])

    @staticmethod
    def from_tokens(tokens: Union[Sequence[int], Tokens]) -> "Prompt":
        """
        Examples:
            >>> prompt = Prompt.from_tokens(Tokens([1, 2, 3]))
        """
        if isinstance(tokens, List):
            tokens = Tokens(tokens)
        return Prompt([tokens])

    def to_json(self) -> Sequence[Mapping[str, Any]]:
        return [_to_prompt_item(item) for item in self.items]


def _to_prompt_item(
    item: Union[str, Image, Tokens, Sequence[int]]
) -> Mapping[str, Any]:
    if isinstance(item, str):
        return {"type": "text", "data": item}
    elif isinstance(item, List):
        return {"type": "token_ids", "data": item}
    elif hasattr(item, "to_json"):
        return item.to_json()
    else:
        raise ValueError(
            "The item in the prompt is not valid. Try either a string or an Image."
        )


def _to_serializable_prompt(
    prompt, at_least_one_token=False
) -> Union[str, Sequence[Mapping[str, str]]]:
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

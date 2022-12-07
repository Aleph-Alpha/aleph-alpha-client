from typing import Any, Dict, NamedTuple, Optional, Sequence


class TokenizationRequest(NamedTuple):
    """Describes a tokenization request.

    Parameters
        prompt (str)
            The text prompt which should be converted into tokens

        tokens (bool)
            True to extract text-tokens

        token_ids (bool)
            True to extract token-ids

    Returns
        TokenizationResponse

    Examples:
        >>> request = TokenizationRequest(prompt="This is an example.", tokens=True, token_ids=True)
    """

    prompt: str
    tokens: bool
    token_ids: bool

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        return payload


class TokenizationResponse(NamedTuple):
    tokens: Optional[Sequence[str]] = None
    token_ids: Optional[Sequence[int]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TokenizationResponse":
        return TokenizationResponse(**json)

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TokenizationRequest:
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

    def to_json(self) -> Mapping[str, Any]:
        return self._asdict()

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizationResponse:
    tokens: Optional[Sequence[str]] = None
    token_ids: Optional[Sequence[int]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TokenizationResponse":
        return TokenizationResponse(
            tokens=json.get("tokens"), token_ids=json.get("token_ids")
        )

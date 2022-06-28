from typing import Any, Dict, NamedTuple, Optional, Sequence


class TokenizationRequest(NamedTuple):
    prompt: str
    tokens: bool
    token_ids: bool


class TokenizationResponse(NamedTuple):
    tokens: Optional[Sequence[str]] = None
    token_ids: Optional[Sequence[int]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TokenizationResponse":
        return TokenizationResponse(**json)
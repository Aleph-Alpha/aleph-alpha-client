from typing import Any, Dict, NamedTuple, Optional, Sequence


class TokenizationRequest(NamedTuple):
    prompt: str
    tokens: bool
    token_ids: bool

    def render_as_body(self, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
        }

class TokenizationResponse(NamedTuple):
    tokens: Optional[Sequence[str]] = None
    token_ids: Optional[Sequence[int]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TokenizationResponse":
        return TokenizationResponse(**json)
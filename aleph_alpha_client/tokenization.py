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
    """
    prompt: str
    tokens: bool
    token_ids: bool


class TokenizationResponse(NamedTuple):
    tokens: Optional[Sequence[str]] = None
    token_ids: Optional[Sequence[int]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TokenizationResponse":
        return TokenizationResponse(**json)
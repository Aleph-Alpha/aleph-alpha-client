from typing import Any, Dict, List, NamedTuple, Optional, Sequence


class DetokenizationRequest(NamedTuple):
    """Describes a detokenization request.

    Parameters
        token_ids (Sequence[int])
            Ids of the tokens for which the text should be returned.

    Examples:
        >>> DetokenizationRequest(token_ids=[1730, 387, 300, 4377, 17])
    """

    token_ids: Sequence[int]

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        return payload


class DetokenizationResponse(NamedTuple):
    result: str

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "DetokenizationResponse":
        return DetokenizationResponse(**json)

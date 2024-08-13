from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class DetokenizationRequest:
    """Describes a detokenization request.

    Parameters
        token_ids (Sequence[int])
            Ids of the tokens for which the text should be returned.

    Examples:
        >>> DetokenizationRequest(token_ids=[1730, 387, 300, 4377, 17])
    """

    token_ids: Sequence[int]

    def to_json(self) -> Mapping[str, Any]:
        payload = self._asdict()
        return payload

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetokenizationResponse:
    result: str

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "DetokenizationResponse":
        return DetokenizationResponse(result=json["result"])

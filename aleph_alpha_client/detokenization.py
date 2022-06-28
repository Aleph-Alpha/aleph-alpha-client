from typing import Any, Dict, List, NamedTuple, Optional, Sequence


class DetokenizationRequest(NamedTuple):
    token_ids: Sequence[int]


class DetokenizationResponse(NamedTuple):
    result: Sequence[str]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "DetokenizationResponse":
        return DetokenizationResponse(**json)

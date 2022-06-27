from typing import Any, Dict, List, NamedTuple, Optional, Sequence


class DetokenizationRequest(NamedTuple):
    token_ids: Sequence[int]

    def render_as_body(self, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "token_ids": self.token_ids,
        }


class DetokenizationResponse(NamedTuple):
    result: Sequence[str]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "DetokenizationResponse":
        return DetokenizationResponse(**json)

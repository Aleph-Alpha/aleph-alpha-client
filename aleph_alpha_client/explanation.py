from typing import Any, List, Dict, NamedTuple, Optional
from aleph_alpha_client.prompt import Prompt


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    suppression_factor: float
    conceptual_suppression_threshold: Optional[float] = None
    normalize: Optional[bool] = None
    square_outputs: Optional[bool] = None
    prompt_explain_indices: Optional[List[int]] = None

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        payload["prompt"] = self.prompt.to_json()
        return payload


class ExplanationResponse(NamedTuple):
    model_version: str
    result: List[Any]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "ExplanationResponse":
        return ExplanationResponse(
            model_version=json["model_version"],
            result=json["result"],
        )

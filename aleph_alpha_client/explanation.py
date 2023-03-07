from typing import Any, Generic, List, Dict, NamedTuple, Optional, TypeVar, Union
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


class Explanation2Request(NamedTuple):
    prompt: Prompt
    target: str

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        payload["prompt"] = self.prompt.to_json()
        return payload


class TextImportance(NamedTuple):
    start: int
    length: int
    score: float

    @staticmethod
    def from_json(score: Any) -> "TextImportance":
        return TextImportance(
            start=score["start"],
            length=score["length"],
            score=score["score"],
        )


class TargetImportance(NamedTuple):
    start: int
    length: int
    score: float

    @staticmethod
    def from_json(score: Any) -> "TextImportance":
        return TextImportance(
            start=score["start"],
            length=score["length"],
            score=score["score"],
        )


class TokenImportance(NamedTuple):
    start: int
    length: int
    score: float

    @staticmethod
    def from_json(score: Any) -> "TextImportance":
        return TextImportance(
            start=score["start"],
            length=score["length"],
            score=score["score"],
        )


class PromptItemExplanation(NamedTuple):
    scores: List[Union[TextImportance, TargetImportance, TokenImportance]]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "PromptItemExplanation":
        if item["type"] == "text":
            return PromptItemExplanation(
                scores=[
                    TextImportance.from_json(importance)
                    for importance in item["scores"]
                ]
            )
        elif item["type"] == "target":
            return PromptItemExplanation(
                scores=[
                    TargetImportance.from_json(importance)
                    for importance in item["scores"]
                ]
            )
        elif item["type"] == "token_ids":
            return PromptItemExplanation(
                scores=[
                    TokenImportance.from_json(importance)
                    for importance in item["scores"]
                ]
            )
        else:
            raise NotImplementedError("Unsupported explanation type")


class Explanation(NamedTuple):
    target: str
    items: List[PromptItemExplanation]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "Explanation":
        return Explanation(
            target=json["target"],
            items=[PromptItemExplanation.from_json(item) for item in json["items"]],
        )


class Explanation2Response(NamedTuple):
    model_version: str
    explanations: List[Explanation]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "Explanation2Response":
        return Explanation2Response(
            model_version=json["model_version"],
            explanations=[
                Explanation.from_json(explanation)
                for explanation in json["explanations"]
            ],
        )

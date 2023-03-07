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


class TextScore(NamedTuple):
    start: int
    length: int
    score: float

    @staticmethod
    def from_json(score: Any) -> "TextScore":
        return TextScore(
            start=score["start"],
            length=score["length"],
            score=score["score"],
        )


class TargetScore(NamedTuple):
    start: int
    length: int
    score: float

    @staticmethod
    def from_json(score: Any) -> "TargetScore":
        return TargetScore(
            start=score["start"],
            length=score["length"],
            score=score["score"],
        )


class TokenScore(NamedTuple):
    score: float

    @staticmethod
    def from_json(score: Any) -> "TokenScore":
        return TokenScore(
            score=score,
        )


class TextPromptItemExplanation(NamedTuple):
    scores: List[TextScore]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TextPromptItemExplanation":
        return TextPromptItemExplanation(
            scores=[TextScore.from_json(score) for score in item["scores"]]
        )


class TargetPromptItemExplanation(NamedTuple):
    scores: List[TargetScore]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TargetPromptItemExplanation":
        return TargetPromptItemExplanation(
            scores=[TargetScore.from_json(score) for score in item["scores"]]
        )


class TokenPromptItemExplanation(NamedTuple):
    scores: List[TokenScore]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TokenPromptItemExplanation":
        return TokenPromptItemExplanation(
            scores=[TokenScore.from_json(score) for score in item["scores"]]
        )


class Explanation(NamedTuple):
    target: str
    items: List[
        Union[
            TextPromptItemExplanation,
            TargetPromptItemExplanation,
            TokenPromptItemExplanation,
        ]
    ]

    def prompt_item_from_json(
        item: Any,
    ) -> Union[
        TextPromptItemExplanation,
        TargetPromptItemExplanation,
        TokenPromptItemExplanation,
    ]:
        if item["type"] == "text":
            return TextPromptItemExplanation.from_json(item)
        elif item["type"] == "target":
            return TargetPromptItemExplanation.from_json(item)
        elif item["type"] == "token_ids":
            return TokenPromptItemExplanation.from_json(item)
        else:
            raise NotImplementedError("Unsupported explanation type")

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "Explanation":
        return Explanation(
            target=json["target"],
            items=[Explanation.prompt_item_from_json(item) for item in json["items"]],
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

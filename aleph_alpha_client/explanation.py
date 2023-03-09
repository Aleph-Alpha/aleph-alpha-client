from enum import Enum
from typing import (
    Any,
    Generic,
    List,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)
from aleph_alpha_client.prompt import Prompt


class ExplanationGranularity(Enum):
    """
    Available types of explanation granularity for text or image prompt items

    Token:
        Explain token by token
    Word:
        Explain word by word. Consecutive whitespace characters define a word boundary.
    """

    Token = "token"
    Word = "word"

    def to_json(self) -> Mapping[str, Any]:
        return {"type": self.value}


class ExplanationPostprocessing(Enum):
    """
    Available types of explanation postprocessing.

    Square:
        Square each score
    Absolute:
        Take the absolute value of each score
    """

    Square = "square"
    Absolute = "absolute"

    def to_json(self) -> str:
        return self.value


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    granularity: Optional[ExplanationGranularity] = None
    control_factor: Optional[float] = None
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = None
    postprocessing: Optional[ExplanationPostprocessing] = None
    normalize: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        payload = {k: v for k, v in self._asdict().items() if v is not None}
        payload["prompt"] = self.prompt.to_json()
        if self.granularity is not None:
            payload["granularity"] = self.granularity.to_json()
        if self.postprocessing is not None:
            payload["postprocessing"] = self.postprocessing.to_json()

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


class ImageScore(NamedTuple):
    left: float
    top: float
    width: float
    height: float
    score: float

    @staticmethod
    def from_json(score: Any) -> "ImageScore":
        return ImageScore(
            left=score["rect"]["left"],
            top=score["rect"]["top"],
            width=score["rect"]["width"],
            height=score["rect"]["height"],
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


class ImagePromptItemExplanation(NamedTuple):
    scores: List[ImageScore]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "ImagePromptItemExplanation":
        return ImagePromptItemExplanation(
            scores=[ImageScore.from_json(score) for score in item["scores"]]
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
            ImagePromptItemExplanation,
        ]
    ]

    def prompt_item_from_json(
        item: Any,
    ) -> Union[
        TextPromptItemExplanation,
        ImagePromptItemExplanation,
        TargetPromptItemExplanation,
        TokenPromptItemExplanation,
    ]:
        if item["type"] == "text":
            return TextPromptItemExplanation.from_json(item)
        elif item["type"] == "target":
            return TargetPromptItemExplanation.from_json(item)
        elif item["type"] == "image":
            return ImagePromptItemExplanation.from_json(item)
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


class ExplanationResponse(NamedTuple):
    model_version: str
    explanations: List[Explanation]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "ExplanationResponse":
        return ExplanationResponse(
            model_version=json["model_version"],
            explanations=[
                Explanation.from_json(explanation)
                for explanation in json["explanations"]
            ],
        )

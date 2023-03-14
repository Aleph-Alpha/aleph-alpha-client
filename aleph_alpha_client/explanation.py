from enum import Enum
from typing import (
    Any,
    Generic,
    List,
    Dict,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)
from aleph_alpha_client.prompt import Prompt


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


class CustomGranularity(NamedTuple):
    delimiter: str

    def to_json(self) -> Mapping[str, Any]:
        return {"type": "custom", "delimiter": self.delimiter}


ExplanationGranularity = Union[
    Literal["token"],
    Literal["word"],
    Literal["sentence"],
    Literal["paragraph"],
    CustomGranularity,
]


def granularity_to_json(granularity: ExplanationGranularity) -> Mapping[str, Any]:
    if isinstance(granularity, str):
        return {"type": granularity}

    return granularity.to_json()


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    contextual_control_threshold: Optional[float] = None
    control_factor: Optional[float] = None
    control_log_additive: Optional[bool] = None
    granularity: Optional[ExplanationGranularity] = None
    postprocessing: Optional[ExplanationPostprocessing] = None
    normalize: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": self.prompt.to_json(),
            "target": self.target,
            "contextual_control_threshold": self.contextual_control_threshold,
        }
        if self.control_factor is not None:
            payload["control_factor"] = self.control_factor
        if self.control_log_additive is not None:
            payload["control_log_additive"] = self.control_log_additive
        if self.granularity is not None:
            payload["granulariy"] = granularity_to_json(self.granularity)
        if self.postprocessing is not None:
            payload["postprocessing"] = self.postprocessing.to_json()
        if self.normalize is not None:
            payload["normalize"] = self.normalize

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

from enum import Enum
from typing import (
    Any,
    List,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Union,
)

# Import Literal with Python 3.7 fallback
from typing_extensions import Literal

from aleph_alpha_client.prompt import ControlTokenOverlap, Image, Prompt, PromptItem


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


PromptGranularity = Union[
    Literal["token"],
    Literal["word"],
    Literal["sentence"],
    Literal["paragraph"],
    CustomGranularity,
]


def prompt_granularity_to_json(
    prompt_granularity: PromptGranularity,
) -> Mapping[str, Any]:
    if isinstance(prompt_granularity, str):
        return {"type": prompt_granularity}

    return prompt_granularity.to_json()


class TargetGranularity(Enum):
    """
    How many explanations should be returned in the output.

    Complete:
        Return one explanation for the entire target. Helpful in many cases to determine which parts of the prompt contribute overall to the given completion.
    Token:
        Return one explanation for each token in the target.
    """

    Complete = "complete"
    Token = "token"

    def to_json(self) -> str:
        return self.value


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    contextual_control_threshold: Optional[float] = None
    control_factor: Optional[float] = None
    control_token_overlap: Optional[ControlTokenOverlap] = None
    control_log_additive: Optional[bool] = None
    prompt_granularity: Optional[PromptGranularity] = None
    target_granularity: Optional[TargetGranularity] = None
    postprocessing: Optional[ExplanationPostprocessing] = None
    normalize: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": self.prompt.to_json(),
            "target": self.target,
        }
        if self.contextual_control_threshold is not None:
            payload["contextual_control_threshold"] = self.contextual_control_threshold
        if self.control_token_overlap is not None:
            payload["control_token_overlap"] = self.control_token_overlap.to_json()
        if self.postprocessing is not None:
            payload["postprocessing"] = self.postprocessing.to_json()
        if self.control_log_additive is not None:
            payload["control_log_additive"] = self.control_log_additive
        if self.prompt_granularity is not None:
            payload["prompt_granularity"] = prompt_granularity_to_json(
                self.prompt_granularity
            )
        if self.target_granularity is not None:
            payload["target_granularity"] = self.target_granularity.to_json()
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

    def in_pixels(self, prompt_item: PromptItem) -> "ImagePromptItemExplanation":
        if not isinstance(prompt_item, Image):
            raise ValueError
        (original_image_width, original_image_height) = prompt_item.dimensions()
        return ImagePromptItemExplanation(
            [
                ImageScore(
                    left=int(score.left * original_image_width),
                    width=int(score.width * original_image_width),
                    top=int(score.top * original_image_height),
                    height=int(score.height * original_image_height),
                    score=score.score,
                )
                for score in self.scores
            ]
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

    def with_image_prompt_items_in_pixels(self, prompt: Prompt) -> "Explanation":
        return Explanation(
            target=self.target,
            items=[
                item.in_pixels(prompt.items[item_index])
                if isinstance(item, ImagePromptItemExplanation)
                else item
                for item_index, item in enumerate(self.items)
            ],
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

    def with_image_prompt_items_in_pixels(
        self, prompt: Prompt
    ) -> "ExplanationResponse":
        mapped_explanations = [
            explanation.with_image_prompt_items_in_pixels(prompt)
            for explanation in self.explanations
        ]
        return ExplanationResponse(self.model_version, mapped_explanations)

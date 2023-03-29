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
    """
    Allows for passing a custom delimiter to determine the granularity to
    to explain the prompt by. The text of the prompt will be split by the
    delimiter you provide.

    Parameters:
        delimiter (str, required):
            String to split the text in the prompt by for generating
            explanations for your prompt.
    """

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
    """
    Describes an Explanation request you want to make agains the API.

    Parameters:
        prompt (Prompt, required)
            Prompt you want to generate explanations for a target completion.
        target (str, required)
            The completion string to be explained based on model probabilities.
        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.
        control_factor (float, default None):
            The amount to adjust model attention by.
            For Explanation, you want to supress attention, and the API will default to 0.1.
            Values between 0 and 1 will supress attention.
            A value of 1 will have no effect.
            Values above 1 will increase attention.
        control_token_overlap (ControlTokenOverlap, default None)
            What to do if a control partially overlaps with a text or image token.
            If set to "partial", the factor will be adjusted proportionally with the amount
            of the token it overlaps. So a factor of 2.0 of a control that only covers 2 of
            4 token characters, would be adjusted to 1.5.
            If set to "complete", the full factor will be applied as long as the control
            overlaps with the token at all.
        control_log_additive (bool, default None)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor
            If None, the API will default to True
        prompt_granularity (PromptGranularity, default None)
            At which granularity should the target be explained in terms of the prompt.
            If you choose, for example, "sentence" then we report the importance score of each
            sentence in the prompt towards generating the target output.

            If you do not choose a granularity then we will try to find the granularity that
            brings you closest to around 30 explanations. For large documents, this would likely
            be sentences. For short prompts this might be individual words or even tokens.

            If you choose a custom granularity then you must provide a custom delimiter. We then
            split your prompt by that delimiter. This might be helpful if you are using few-shot
            prompts that contain stop sequences.

            For image prompt items, the granularities determine into how many tiles we divide
            the image for the explanation.
            "token" -> 12x12
            "word" -> 6x6
            "sentence" -> 3x3
            "paragraph" -> 1
        target_granularity (TargetGranularity, default None)
            How many explanations should be returned in the output.

            "complete" -> Return one explanation for the entire target. Helpful in many cases to determine which parts of the prompt contribute overall to the given completion.
            "token" -> Return one explanation for each token in the target.

            If None, API will default to "complete"
        postprocessing (ExplanationPostprocessing, default None)
            Optionally apply postprocessing to the difference in cross entropy scores for each token.
            "none": Apply no postprocessing.
            "absolute": Return the absolute value of each value.
            "square": Square each value
        normalize (bool, default None)
            Return normalized scores. Minimum score becomes 0 and maximum score becomes 1. Applied after any postprocessing
    """

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

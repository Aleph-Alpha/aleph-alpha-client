from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    List,
    Dict,
    Mapping,
    Optional,
    Union,
)

from aleph_alpha_client import Text

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


@dataclass(frozen=True)
class CustomGranularity:
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


class PromptGranularity(Enum):
    Token = "token"
    Word = "word"
    Sentence = "sentence"
    Paragraph = "paragraph"

    def to_json(self) -> Mapping[str, Any]:
        return {"type": self.value}


def prompt_granularity_to_json(
    prompt_granularity: Union[PromptGranularity, str, CustomGranularity],
) -> Mapping[str, Any]:
    # we allow str for backwards compatibility
    # This was previously possible because PromptGranularity was not an Enum
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


@dataclass(frozen=True)
class ExplanationRequest:
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
        prompt_granularity (Union[PromptGranularity, str, CustomGranularity], default None)
            At which granularity should the target be explained in terms of the prompt.
            If you choose, for example, "sentence" then we report the importance score of each
            sentence in the prompt towards generating the target output.

            If you do not choose a granularity then we will try to find the granularity that
            brings you closest to around 30 explanations. For large documents, this would likely
            be sentences. For short prompts this might be individual words or even tokens.

            If you choose a custom granularity then you must provide a custom delimiter. We then
            split your prompt by that delimiter. This might be helpful if you are using few-shot
            prompts that contain stop sequences.

            We currently support providing the prompt_granularity as PromptGranularity (recommended)
            or CustomGranularity (if needed) or str (deprecated). Note that supplying plain strings
            only makes sense if you choose one of the values defined in the PromptGranularity enum.
            All other strings will be rejected by the API. In future versions we might cut support
            for plain str values.

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
    prompt_granularity: Optional[Union[PromptGranularity, str, CustomGranularity]] = (
        None
    )
    target_granularity: Optional[TargetGranularity] = None
    postprocessing: Optional[ExplanationPostprocessing] = None
    normalize: Optional[bool] = None

    def to_json(self) -> Mapping[str, Any]:
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
        if self.control_factor is not None:
            payload["control_factor"] = self.control_factor

        return payload


@dataclass(frozen=True)
class TextScore:
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


@dataclass(frozen=True)
class TextScoreWithRaw:
    start: int
    length: int
    score: float
    text: str

    @staticmethod
    def from_text_score(score: TextScore, prompt: Text) -> "TextScoreWithRaw":
        return TextScoreWithRaw(
            start=score.start,
            length=score.length,
            score=score.score,
            text=prompt.text[score.start : score.start + score.length],
        )


@dataclass(frozen=True)
class ImageScore:
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


@dataclass(frozen=True)
class TargetScore:
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


@dataclass(frozen=True)
class TargetScoreWithRaw:
    start: int
    length: int
    score: float
    text: str

    @staticmethod
    def from_target_score(score: TargetScore, target: str) -> "TargetScoreWithRaw":
        return TargetScoreWithRaw(
            start=score.start,
            length=score.length,
            score=score.score,
            text=target[score.start : score.start + score.length],
        )


@dataclass(frozen=True)
class TokenScore:
    score: float

    @staticmethod
    def from_json(score: Any) -> "TokenScore":
        return TokenScore(
            score=score,
        )


@dataclass(frozen=True)
class ImagePromptItemExplanation:
    """
    Explains the importance of an image prompt item.
    The amount of items in the "scores" array depends on the granularity setting.
    Each score object contains the top-left corner of a rectangular area in the image prompt.
    The coordinates are all between 0 and 1 in terms of the total image size
    """

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


@dataclass(frozen=True)
class TextPromptItemExplanation:
    """
    Explains the importance of a text prompt item.
    The amount of items in the "scores" array depends on the granularity setting.
    Each score object contains an inclusive start character and a length of the substring plus
    a floating point score value.
    """

    scores: List[Union[TextScore, TextScoreWithRaw]]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TextPromptItemExplanation":
        return TextPromptItemExplanation(
            scores=[TextScore.from_json(score) for score in item["scores"]]
        )

    def with_text(self, prompt: Text) -> "TextPromptItemExplanation":
        return TextPromptItemExplanation(
            scores=[
                (
                    TextScoreWithRaw.from_text_score(score, prompt)
                    if isinstance(score, TextScore)
                    else score
                )
                for score in self.scores
            ]
        )


@dataclass(frozen=True)
class TargetPromptItemExplanation:
    """
    Explains the importance of text in the target string that came before the currently
    to-be-explained target token. The amount of items in the "scores" array depends on the
    granularity setting.
    Each score object contains an inclusive start character and a length of the substring plus
    a floating point score value.
    """

    scores: List[Union[TargetScore, TargetScoreWithRaw]]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TargetPromptItemExplanation":
        return TargetPromptItemExplanation(
            scores=[TargetScore.from_json(score) for score in item["scores"]]
        )

    def with_text(self, prompt: str) -> "TargetPromptItemExplanation":
        return TargetPromptItemExplanation(
            scores=[
                (
                    TargetScoreWithRaw.from_target_score(score, prompt)
                    if isinstance(score, TargetScore)
                    else score
                )
                for score in self.scores
            ]
        )


@dataclass(frozen=True)
class TokenPromptItemExplanation:
    """Explains the importance of a request prompt item of type "token_ids".
    Will contain one floating point importance value for each token in the same order as in the original prompt.
    """

    scores: List[TokenScore]

    @staticmethod
    def from_json(item: Dict[str, Any]) -> "TokenPromptItemExplanation":
        return TokenPromptItemExplanation(
            scores=[TokenScore.from_json(score) for score in item["scores"]]
        )


@dataclass(frozen=True)
class Explanation:
    """
    Explanations for a given portion of the target.

    Parameters:
        target (str, required)
            If target_granularity was set to "complete", then this will be the entire target. If it was set to "token", this will be a single target token.
        items (List[Union[TextPromptItemExplanation, TargetPromptItemExplanation, TokenPromptItemExplanation, ImagePromptItemExplanation], required)
            Contains one item for each prompt item (in order), and the last item refers to the target.
    """

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
                (
                    item.in_pixels(prompt.items[item_index])
                    if isinstance(item, ImagePromptItemExplanation)
                    else item
                )
                for item_index, item in enumerate(self.items)
            ],
        )

    def with_text_from_prompt(self, prompt: Prompt, target: str) -> "Explanation":
        items: List[
            Union[
                TextPromptItemExplanation,
                ImagePromptItemExplanation,
                TargetPromptItemExplanation,
                TokenPromptItemExplanation,
            ]
        ] = []
        for item_index, item in enumerate(self.items):
            if isinstance(item, TextPromptItemExplanation):
                # separate variable to fix linting error
                prompt_item = prompt.items[item_index]
                if isinstance(prompt_item, Text):
                    items.append(item.with_text(prompt_item))
                else:
                    items.append(item)
            elif isinstance(item, TargetPromptItemExplanation):
                items.append(item.with_text(target))
            else:
                items.append(item)
        return Explanation(
            target=self.target,
            items=items,
        )


@dataclass(frozen=True)
class ExplanationResponse:
    """
    The top-level response data structure that will be returned from an explanation request.

    Parameters:
        model_version (str, required)
            Version of the model used to generate the explanation.
        explanations (List[Explanation], required)
            This array will contain one explanation object for each portion of the target.
    """

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

    def with_text_from_prompt(
        self, request: ExplanationRequest
    ) -> "ExplanationResponse":
        mapped_explanations = [
            explanation.with_text_from_prompt(request.prompt, request.target)
            for explanation in self.explanations
        ]
        return ExplanationResponse(self.model_version, mapped_explanations)

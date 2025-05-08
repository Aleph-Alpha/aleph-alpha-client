from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranslationRequest:
    """Request for translation.

    Parameters:
        model (string, required):
            The name of the model to be used for the translation.

        source (string, required):
            The input text to be translated.

        target_language (string, required):
            The desired target language into which the input text should be translated. The language
            must be specified using ISO 639 (and RFC 1766) language codes such as "en" for English,
            "de" for German, "fr" for French, etc. For a list of supported languages, refer to the
            `/languages` endpoint.
    """

    model: str
    source: str
    target_language: str

    def to_json(self) -> dict:
        """Convert the request to a JSON-serializable dictionary."""
        return {
            "model": self.model,
            "source": self.source,
            "target_language": self.target_language,
        }


@dataclass
class TranslationSegment:
    """A segment of translated text with its source and quality score.

    Parameters:
        source (string):
            The input text to be translated.
        translation (string):
            The translated output text of the segment.
        score (float):
            Estimate for the overall quality of the translation on a scale of 0 to 1.
    """

    source: str
    translation: str
    score: float


@dataclass
class TranslationResponse:
    """Response from a translation request.

    Parameters:
        translation (string):
            The complete translated output text
        score (float):
            Estimate for the overall quality of the translation on a scale of 0 to 1.
        segments (Optional[List[TranslationSegment]]):
            List of translated segments. May be None if no segments are returned.
        num_tokens_prompt_total (int):
            Total number of tokens in the prompt. May be zero as the current translation
            models don't expose tokens.
        num_tokens_generated (int):
            Total number of tokens generated. May be zero as the current translation
            models don't expose tokens.
    """

    translation: str
    score: float
    segments: Optional[List[TranslationSegment]]
    num_tokens_prompt_total: int
    num_tokens_generated: int

    @classmethod
    def from_json(cls, json: dict) -> "TranslationResponse":
        """Create a TranslationResponse from a JSON dictionary."""
        segments = None
        if json.get("segments") is not None:
            segments = [
                TranslationSegment(
                    source=segment["source"],
                    translation=segment["translation"],
                    score=segment["score"],
                )
                for segment in json["segments"]
            ]

        return cls(
            translation=json["translation"],
            score=json["score"],
            segments=segments,
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
            num_tokens_generated=json["num_tokens_generated"],
        )

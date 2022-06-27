from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import _to_serializable_prompt


class EvaluationRequest(NamedTuple):
    """
    Evaluates the model's likelihood to produce a completion given a prompt.

    Parameters:
        prompt (str, optional, default ""):
            The text to be completed. Unconditional completion can be used with an empty string (default). The prompt may contain a zero shot or few shot task.

        completion_expected (str, required):
            The ground truth completion expected to be produced given the prompt.
    """

    prompt: Sequence[Union[str, ImagePrompt]]
    completion_expected: str

    def render_as_body(self, model: str, hosting=Optional[str]) -> dict:
        return {
            "model": model,
            "hosting": hosting,
            "prompt": _to_serializable_prompt(self.prompt),
            "completion_expected": self.completion_expected,
        }


class EvaluationResponse(NamedTuple):
    model_version: str
    message: Optional[str]
    result: Dict[str, Any]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "EvaluationResponse":
        return EvaluationResponse(
            model_version=json["model_version"],
            result=json["result"],
            message=json.get("message"),
        )

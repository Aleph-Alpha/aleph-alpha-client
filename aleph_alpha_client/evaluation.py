from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
)
from aleph_alpha_client.prompt import Prompt


class EvaluationRequest(NamedTuple):
    """
    Evaluates the model's likelihood to produce a completion given a prompt.

    Parameters:
        prompt (str, optional, default ""):
            The text to be completed. Unconditional completion can be used with an empty string (default). The prompt may contain a zero shot or few shot task.

        completion_expected (str, required):
            The ground truth completion expected to be produced given the prompt.

    Examples:
        >>> request = EvaluationRequest(prompt=Prompt.from_text("The api works"), completion_expected=" well")
    """

    prompt: Prompt
    completion_expected: str

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        payload["prompt"] = self.prompt.to_json()
        return payload


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

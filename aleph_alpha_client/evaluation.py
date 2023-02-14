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

        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.

        control_log_additive (bool, default True)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor

    Examples:
        >>> request = EvaluationRequest(prompt=Prompt.from_text("The api works"), completion_expected=" well")
    """

    prompt: Prompt
    completion_expected: str
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = True

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

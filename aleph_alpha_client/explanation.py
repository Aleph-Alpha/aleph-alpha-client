from typing import List, NamedTuple, Optional, Union
from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import _to_prompt_item


class ExplanationRequest(NamedTuple):
    prompt: Union[str, List[Union[str, ImagePrompt]]]
    target: str
    directional: bool
    suppression_factor: float
    conceptual_suppression_threshold: Optional[float] = None

    def render_as_body(self, model: str, hosting=Optional[str]) -> dict:

        if isinstance(self.prompt, list):
            prompt = [_to_prompt_item(item) for item in self.prompt]
        else:
            prompt = _to_prompt_item(prompt)
        return {
            "model": model,
            "prompt": prompt,
            "target": self.target,
            "suppression_factor": self.suppression_factor,
            "directional": self.directional,
            "conceptual_suppression_threshold": self.conceptual_suppression_threshold,
        }

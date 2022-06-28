from typing import List, NamedTuple, Optional, Union
from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import _to_prompt_item


class ExplanationRequest(NamedTuple):
    prompt: List[Union[str, ImagePrompt]]
    target: str
    directional: bool
    suppression_factor: float
    conceptual_suppression_threshold: Optional[float] = None

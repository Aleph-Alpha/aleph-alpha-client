from typing import List, NamedTuple, Optional
from aleph_alpha_client.prompt import Prompt


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    suppression_factor: float
    conceptual_suppression_threshold: Optional[float] = None
    normalize: Optional[bool] = None
    square_outputs: Optional[bool] = None
    prompt_explain_indices: Optional[List[int]] = None

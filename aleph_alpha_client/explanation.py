from typing import List, NamedTuple, Optional, Union
from aleph_alpha_client.prompt import Prompt


class ExplanationRequest(NamedTuple):
    prompt: Prompt
    target: str
    directional: bool
    suppression_factor: float
    conceptual_suppression_threshold: Optional[float] = None

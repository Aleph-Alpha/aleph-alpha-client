from typing import Any, Dict, List, NamedTuple, Optional, Sequence


class DetokenizationRequest(NamedTuple):
    """Describes a detokenization request.
    
    Parameters
        token_ids (Sequence[int])
            Ids of the tokens for which the text should be returned.
    """
    token_ids: Sequence[int]


class DetokenizationResponse(NamedTuple):
    result: Sequence[str]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "DetokenizationResponse":
        return DetokenizationResponse(**json)

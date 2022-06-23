from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import _to_prompt_item


class EmbeddingRequest(NamedTuple):
    """
    Embeds a text and returns vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).

    Parameters:
        prompt
            The text and/or image(s) to be embedded.

        layers
            A list of layer indices from which to return embeddings.
                * Index 0 corresponds to the word embeddings used as input to the first transformer layer
                * Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
                * Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last layer etc.

        pooling
            Pooling operation to use.
            Pooling operations include:
                * mean: aggregate token embeddings across the sequence dimension using an average
                * max: aggregate token embeddings across the sequence dimension using a maximum
                * last_token: just use the last token
                * abs_max: aggregate token embeddings across the sequence dimension using a maximum of absolute values

        type
            Type of the embedding (e.g. symmetric or asymmetric)

        tokens
            Flag indicating whether the tokenized prompt is to be returned (True) or not (False)

    """
    prompt: List[Union[str, ImagePrompt]]
    layers: List[int]
    pooling: List[str]
    type: Optional[str] = None
    tokens: bool = False

    def render_as_body(self, model: str, hosting=Optional[str]) -> dict:
        return {
            "model": model,
            "prompt": [_to_prompt_item(item) for item in self.prompt],
            "layers": self.layers,
            "pooling": self.pooling,
            "type": self.type,
            "tokens": self.tokens
        }

class EmbeddingResponse(NamedTuple):
    model_version: str
    embeddings: Optional[Dict[Tuple[str, str], List[float]]]
    tokens: Optional[List[str]]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "EmbeddingResponse":
        return EmbeddingResponse(
            model_version=json["model_version"],
            embeddings={(layer, pooling): embedding for layer, pooling_dict in json["embeddings"].items() for pooling, embedding in pooling_dict.items()},
            tokens=json.get("tokens"))
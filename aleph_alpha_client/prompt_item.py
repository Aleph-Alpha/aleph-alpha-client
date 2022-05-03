from typing import Dict, Union

from aleph_alpha_client.image import ImagePrompt


def _to_prompt_item(item: Union[str, ImagePrompt]) -> Dict[str, str]:
    if isinstance(item, str):
        return {"type": "text", "data": item}
    if hasattr(item, "_to_prompt_item"):
        return item._to_prompt_item()
    else:
        raise ValueError(
            "The item in the prompt is not valid. Try either a string or an Image."
        )

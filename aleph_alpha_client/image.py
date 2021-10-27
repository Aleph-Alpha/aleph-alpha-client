import base64
from typing import Dict
import requests


class ImagePrompt:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.
    """

    def __init__(self, base_64: str):
        # We use a base_64 reperesentation, because we want to embed the image
        # into a prompt send in JSON.
        self.base_64 = base_64

    @classmethod
    def from_url(cls, url: str):
        """
        Downloads a file and prepare it to be used in a prompt
        """
        image = base64.b64encode(requests.get(url).content).decode()
        return cls(image)

    @classmethod
    def from_file(cls, path: str):
        """
        Load an image from disk and prepare it to be used in a prompt
        """
        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode()
        return cls(image)

    def _to_prompt_item(self) -> Dict[str, str]:
        """
        A dict if serialized to JSON is suitable as a prompt element
        """
        return {"type": "image", "data": self.base_64}

import base64
from typing import Dict
import requests


class ImagePrompt:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.
    """

    def __init__(self, base_64: str, x: int = None, y: int = None, size: int = None):
        # We use a base_64 reperesentation, because we want to embed the image
        # into a prompt send in JSON.
        self.base_64 = base_64
        self.x = x
        self.y = y
        self.size = size

    @classmethod
    def from_url(cls, url: str, x: int = None, y: int = None, size: int = None):
        """
        Downloads a file and prepare it to be used in a prompt.
        x, y and size are used to crop the image.
        If they are not provided then the image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        image = base64.b64encode(requests.get(url).content).decode()
        return cls(image, x, y, size)

    @classmethod
    def from_file(cls, path: str, x: int = None, y: int = None, size: int = None):
        """
        Load an image from disk and prepare it to be used in a prompt
        x, y and size are used to crop the image.
        If they are not provided then the image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode()
        return cls(image, x, y, size)

    def _to_prompt_item(self) -> Dict[str, str]:
        """
        A dict if serialized to JSON is suitable as a prompt element
        """
        return {
            "type": "image",
            "data": self.base_64,
            "x": self.x,
            "y": self.y,
            "size": self.size,
        }

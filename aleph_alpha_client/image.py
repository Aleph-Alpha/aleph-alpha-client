import base64
from typing import Any, Dict, Optional
import requests


class Cropping:
    """
    Describes a quadratic crop of the file.
    """
    def __init__(self, upper_left_x: int, upper_left_y: int, size: int):
        self.upper_left_x = upper_left_x
        self.upper_left_y = upper_left_y
        self.size = size


class ImagePrompt:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.
    """
    def __init__(
        self,
        base_64: str,
        cropping: Optional[Cropping] = None,
    ):
        # We use a base_64 reperesentation, because we want to embed the image
        # into a prompt send in JSON.
        self.base_64 = base_64
        self.cropping = cropping

    @classmethod
    def from_bytes(cls, bytes: bytes, cropping: Optional[Cropping] = None):
        image = base64.b64encode(bytes).decode()
        return cls(image, cropping)

    @classmethod
    def from_url(cls, url: str):
        """
        Downloads a file and prepare it to be used in a prompt.
        The image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        bytes = requests.get(url).content
        return cls.from_bytes(bytes)

    @classmethod
    def from_url_with_cropping(cls, url: str, upper_left_x: int,
                               upper_left_y: int, crop_size: int):
        """
        Downloads a file and prepare it to be used in a prompt.
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        cropping = Cropping(upper_left_x=upper_left_x,
                            upper_left_y=upper_left_y,
                            size=crop_size)
        bytes = requests.get(url).content
        return cls.from_bytes(bytes, cropping=cropping)

    @classmethod
    def from_file(cls, path: str):
        """
        Load an image from disk and prepare it to be used in a prompt
        If they are not provided then the image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        with open(path, "rb") as f:
            image = f.read()
        return cls.from_bytes(image)

    @classmethod
    def from_file_with_cropping(cls, path: str, upper_left_x: int,
                                upper_left_y: int, crop_size: int):
        """
        Load an image from disk and prepare it to be used in a prompt
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        cropping = Cropping(upper_left_x=upper_left_x,
                            upper_left_y=upper_left_y,
                            size=crop_size)
        with open(path, "rb") as f:
            bytes = f.read()
        return cls.from_bytes(bytes, cropping=cropping)

    def _to_prompt_item(self) -> Dict[str, Any]:
        """
        A dict if serialized to JSON is suitable as a prompt element
        """
        if self.cropping is None:
            return {
                "type": "image",
                "data": self.base_64,
            }
        else:
            return {
                "type": "image",
                "data": self.base_64,
                "x": self.cropping.upper_left_x,
                "y": self.cropping.upper_left_y,
                "size": self.cropping.size,
            }

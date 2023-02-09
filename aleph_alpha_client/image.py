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


class Image:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.

    Examples:
        >>> # You need to choose a model with multimodal capabilities for this example.
        >>> url = "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png"
        >>> image = Image.from_url(url)
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
        return cls.from_bytes(cls._get_url(url))

    @classmethod
    def from_url_with_cropping(
        cls, url: str, upper_left_x: int, upper_left_y: int, crop_size: int
    ):
        """
        Downloads a file and prepare it to be used in a prompt.
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        cropping = Cropping(
            upper_left_x=upper_left_x, upper_left_y=upper_left_y, size=crop_size
        )
        bytes = cls._get_url(url)
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
    def from_file_with_cropping(
        cls, path: str, upper_left_x: int, upper_left_y: int, crop_size: int
    ):
        """
        Load an image from disk and prepare it to be used in a prompt
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        cropping = Cropping(
            upper_left_x=upper_left_x, upper_left_y=upper_left_y, size=crop_size
        )
        with open(path, "rb") as f:
            bytes = f.read()
        return cls.from_bytes(bytes, cropping=cropping)

    @classmethod
    def _get_url(cls, url: str) -> bytes:
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def to_json(self) -> Dict[str, Any]:
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


# For backwards compatibility we still expose this with the old name
ImagePrompt = Image

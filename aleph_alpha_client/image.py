import base64
from typing import Dict, Optional
import requests


class ImagePrompt:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.
    """

    def __init__(
        self,
        base_64: str,
        upper_left_x: Optional[int] = None,
        upper_left_y: Optional[int] = None,
        crop_size: Optional[int] = None,
    ):
        # We use a base_64 reperesentation, because we want to embed the image
        # into a prompt send in JSON.
        self.base_64 = base_64
        self.upper_left_x = upper_left_x
        self.upper_left_y = upper_left_y
        self.crop_size = crop_size

    @classmethod
    def from_url(cls, url: str):
        """
        Downloads a file and prepare it to be used in a prompt.
        The image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        return cls.from_url_with_cropping(url, None, None, None)

    @classmethod
    def from_url_with_cropping(
        cls, url: str, upper_left_x: int, upper_left_y: int, crop_size: int
    ):
        """
        Downloads a file and prepare it to be used in a prompt.
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        image = base64.b64encode(requests.get(url).content).decode()
        return cls(image, upper_left_x, upper_left_y, crop_size)

    @classmethod
    def from_file(cls, path: str):
        """
        Load an image from disk and prepare it to be used in a prompt
        If they are not provided then the image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        return cls.from_file_with_cropping(path, None, None, None)

    @classmethod
    def from_file_with_cropping(
        cls, path: str, upper_left_x: int, upper_left_y: int, crop_size: int
    ):
        """
        Load an image from disk and prepare it to be used in a prompt
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode()
        return cls(image, upper_left_x, upper_left_y, crop_size)

    def _to_prompt_item(self) -> Dict[str, str]:
        """
        A dict if serialized to JSON is suitable as a prompt element
        """
        return {
            "type": "image",
            "data": self.base_64,
            "x": self.upper_left_x,
            "y": self.upper_left_y,
            "size": self.crop_size,
        }

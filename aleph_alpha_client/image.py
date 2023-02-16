import base64
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence
import requests


class Cropping:
    """
    Describes a quadratic crop of the file.
    """

    def __init__(self, upper_left_x: int, upper_left_y: int, size: int):
        self.upper_left_x = upper_left_x
        self.upper_left_y = upper_left_y
        self.size = size


class ImageControl(NamedTuple):
    """
    Attention manipulation for an Image PromptItem.

    All coordinates of the bounding box are logical coordinates (between 0 and 1) and relative to
    the entire image.

    Keep in mind, non-square images are center-cropped by default before going to the model. (You
    can specify a custom cropping if you want.). Since control coordinates are relative to the
    entire image, all or a portion of your control may be outside the "model visible area".

    Parameters:
        left (float, required):
            x-coordinate of top left corner of the control bounding box.
            Must be a value between 0 and 1, where 0 is the left corner and 1 is the right corner.
        top (float, required):
            y-coordinate of top left corner of the control bounding box
            Must be a value between 0 and 1, where 0 is the top pixel row and 1 is the bottom row.
        width (float, required):
            width of the control bounding box
            Must be a value between 0 and 1, where 1 means the full width of the image.
        height (float, required):
            height of the control bounding box
            Must be a value between 0 and 1, where 1 means the full height of the image.
        factor (float, required):
            The amount to adjust model attention by.
            Values between 0 and 1 will supress attention.
            A value of 1 will have no effect.
            Values above 1 will increase attention.
    """

    left: float
    top: float
    width: float
    height: float
    factor: float

    def to_json(self) -> Mapping[str, Any]:
        return {
            "rect": {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            },
            "factor": self.factor,
        }


class Image:
    """
    An image send as part of a prompt to a model. The image is represented as
    base64.

    Note: The models operate on square images. All non-square images are center-cropped
    before going to the model, so portions of the image may not be visible.

    You can supply specific cropping parameters if you like, to choose a different area
    of the image than a center-crop. Or, you can always transform the image yourself to
    a square before sending it.

    Examples:
        >>> # You need to choose a model with multimodal capabilities for this example.
        >>> url = "https://cdn-images-1.medium.com/max/1200/1*HunNdlTmoPj8EKpl-jqvBA.png"
        >>> image = Image.from_url(url)
    """

    def __init__(
        self,
        base_64: str,
        cropping: Optional[Cropping],
        controls: Sequence[ImageControl],
    ):
        # We use a base_64 reperesentation, because we want to embed the image
        # into a prompt send in JSON.
        self.base_64 = base_64
        self.cropping = cropping
        self.controls: Sequence[ImageControl] = controls

    @classmethod
    def from_bytes(
        cls,
        bytes: bytes,
        cropping: Optional[Cropping] = None,
        controls: Optional[Sequence[ImageControl]] = None,
    ):
        image = base64.b64encode(bytes).decode()
        return cls(image, cropping, controls or [])

    @classmethod
    def from_url(cls, url: str, controls: Optional[Sequence[ImageControl]] = None):
        """
        Downloads a file and prepare it to be used in a prompt.
        The image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        return cls.from_bytes(
            cls._get_url(url), cropping=None, controls=controls or None
        )

    @classmethod
    def from_url_with_cropping(
        cls,
        url: str,
        upper_left_x: int,
        upper_left_y: int,
        crop_size: int,
        controls: Optional[Sequence[ImageControl]] = None,
    ):
        """
        Downloads a file and prepare it to be used in a prompt.
        upper_left_x, upper_left_y and crop_size are used to crop the image.
        """
        cropping = Cropping(
            upper_left_x=upper_left_x, upper_left_y=upper_left_y, size=crop_size
        )
        bytes = cls._get_url(url)
        return cls.from_bytes(bytes, cropping=cropping, controls=controls or [])

    @classmethod
    def from_file(cls, path: str, controls: Optional[Sequence[ImageControl]] = None):
        """
        Load an image from disk and prepare it to be used in a prompt
        If they are not provided then the image will be [center cropped](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop)
        """
        with open(path, "rb") as f:
            image = f.read()
        return cls.from_bytes(image, None, controls or [])

    @classmethod
    def from_file_with_cropping(
        cls,
        path: str,
        upper_left_x: int,
        upper_left_y: int,
        crop_size: int,
        controls: Optional[Sequence[ImageControl]] = None,
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
        return cls.from_bytes(bytes, cropping=cropping, controls=controls or None)

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
                "controls": [control.to_json() for control in self.controls],
            }
        else:
            return {
                "type": "image",
                "data": self.base_64,
                "x": self.cropping.upper_left_x,
                "y": self.cropping.upper_left_y,
                "size": self.cropping.size,
                "controls": [control.to_json() for control in self.controls],
            }


# For backwards compatibility we still expose this with the old name
ImagePrompt = Image

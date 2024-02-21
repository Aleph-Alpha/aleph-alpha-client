import base64
from dataclasses import dataclass
import io
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import PIL
import requests
from PIL.Image import Image as PILImage


class ControlTokenOverlap(Enum):
    """
    What to do if a control partially overlaps with a text or image token.

    Partial:
        The factor will be adjusted proportionally with the amount of the token
        it overlaps. So a factor of 2.0 of a control that only covers 2 of 4
        token characters, would be adjusted to 1.5.

    Complete:
        The full factor will be applied as long as the control overlaps with
        the token at all. How many explanations should be returned in the output.
    """

    Partial = "partial"
    Complete = "complete"

    def to_json(self) -> str:
        return self.value


@dataclass(frozen=True)
class TokenControl:
    """
    Used for Attention Manipulation, for a given token index, you can supply
    the factor you want to adjust the attention by.

    Parameters:
        pos (int, required):
            The index of the token in the prompt item that you want to apply
            the factor to.

        factor (float, required):
            The amount to adjust model attention by.
            Values between 0 and 1 will supress attention.
            A value of 1 will have no effect.
            Values above 1 will increase attention.

    Examples:
        >>> Tokens([1, 2, 3], controls=[TokenControl(pos=1, factor=0.5)])
    """

    pos: int
    factor: float

    def to_json(self) -> Mapping[str, Any]:
        return {"index": self.pos, "factor": self.factor}


@dataclass(frozen=True)
class Tokens:
    """
    A list of token ids to be sent as part of a prompt.

    Parameters:
        tokens (List(int), required):
            The tokens you want to be passed to the model as part of your prompt.

        controls (List(TokenControl), optional, default None):
            Used for Attention Manipulation. Provides the ability to change
            attention for given token ids.

    Examples:
        >>> token_ids = Tokens([1, 2, 3], controls=[])
        >>> prompt = Prompt([token_ids])
    """

    tokens: Sequence[int]
    controls: Sequence[TokenControl]

    def to_json(self) -> Mapping[str, Any]:
        """
        Serialize the prompt item to JSON for sending to the API.
        """
        return {
            "type": "token_ids",
            "data": self.tokens,
            "controls": [c.to_json() for c in self.controls],
        }

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "Tokens":
        return Tokens(tokens=json["data"], controls=[])

    @staticmethod
    def from_token_ids(token_ids: Sequence[int]) -> "Tokens":
        return Tokens(token_ids, [])


@dataclass(frozen=True)
class TextControl:
    """
    Attention manipulation for a Text PromptItem.

    Parameters:
        start (int, required):
            Starting character index to apply the factor to.
        length (int, required):
            The amount of characters to apply the factor to.
        factor (float, required):
            The amount to adjust model attention by.
            Values between 0 and 1 will supress attention.
            A value of 1 will have no effect.
            Values above 1 will increase attention.
        token_overlap (ControlTokenOverlap, optional):
            What to do if a control partially overlaps with a text token.

            If set to "partial", the factor will be adjusted proportionally
            with the amount of the token it overlaps. So a factor of 2.0 of a
            control that only covers 2 of 4 token characters, would be adjusted
            to 1.5.

            If set to "complete", the full factor will be applied as long as
            the control overlaps with the token at all.

            If not set, the API will default to "partial".
    """

    start: int
    length: int
    factor: float
    token_overlap: Optional[ControlTokenOverlap] = None

    def to_json(self) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "start": self.start,
            "length": self.length,
            "factor": self.factor,
        }
        if self.token_overlap is not None:
            payload["token_overlap"] = self.token_overlap.to_json()
        return payload


@dataclass(frozen=True)
class Text:
    """
    A Text-prompt including optional controls for attention manipulation.

    Parameters:
        text (str, required):
            The text prompt
        controls (list of TextControl, required):
            A list of TextControls to manilpulate attention when processing the prompt.
            Can be empty if no manipulation is required.

    Examples:
        >>> Text("Hello, World!", controls=[TextControl(start=0, length=5, factor=0.5)])
    """

    text: str
    controls: Sequence[TextControl]

    def to_json(self) -> Mapping[str, Any]:
        return {
            "type": "text",
            "data": self.text,
            "controls": [control.to_json() for control in self.controls],
        }

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "Text":
        return Text.from_text(json["data"])

    @staticmethod
    def from_text(text: str) -> "Text":
        return Text(text, [])


@dataclass(frozen=True)
class Cropping:
    """
    Describes a quadratic crop of the file.
    """

    upper_left_x: int
    upper_left_y: int
    size: int


@dataclass(frozen=True)
class ImageControl:
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
        token_overlap (ControlTokenOverlap, optional):
            What to do if a control partially overlaps with an image token.

            If set to "partial", the factor will be adjusted proportionally
            with the amount of the token it overlaps. So a factor of 2.0 of a
            control that only half of the image "tile", would be adjusted to
            1.5.

            If set to "complete", the full factor will be applied as long as
            the control overlaps with the token at all.

            If not set, the API will default to "partial".
    """

    left: float
    top: float
    width: float
    height: float
    factor: float
    token_overlap: Optional[ControlTokenOverlap] = None

    def to_json(self) -> Mapping[str, Any]:
        payload = {
            "rect": {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            },
            "factor": self.factor,
        }
        if self.token_overlap is not None:
            payload["token_overlap"] = self.token_overlap.to_json()
        return payload


@dataclass(frozen=True)
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

    # We use a base_64 reperesentation, because we want to embed the image
    # into a prompt send in JSON.
    base_64: str
    cropping: Optional[Cropping]
    controls: Sequence[ImageControl]

    @classmethod
    def from_image_source(
        cls,
        image_source: Union[str, Path, bytes],
        controls: Optional[Sequence[ImageControl]] = None,
    ):
        """
        Abstraction on top of the existing methods of image initialization.
        If you are not sure what the exact type of your image, but you know it is either a Path object, URL, a file path,
        or a bytes array, just use the method and we will figure out which of the methods of image initialization to use
        """
        if isinstance(image_source, Path):
            return cls.from_file(path=str(image_source), controls=controls)

        elif isinstance(image_source, str):
            try:
                p = urlparse(image_source)
                if p.scheme:
                    return cls.from_url(url=image_source, controls=controls)
            except Exception as e:
                # we assume that If the string runs into a Exception it isn't not a valid ulr
                pass

            return cls.from_file(path=image_source, controls=controls)

        elif isinstance(image_source, bytes):
            return cls.from_bytes(bytes=image_source, controls=controls)

        else:
            raise TypeError(
                f"The image source: {image_source} should be either Path, str or bytes"
            )

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
    def from_file(
        cls, path: Union[str, Path], controls: Optional[Sequence[ImageControl]] = None
    ):
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

    def to_json(self) -> Mapping[str, Any]:
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

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "Image":
        return Image(base_64=json["data"], cropping=None, controls=[])

    def to_image(self) -> PILImage:
        return PIL.Image.open(io.BytesIO(base64.b64decode(self.base_64)))

    def dimensions(self) -> Tuple[int, int]:
        image = self.to_image()
        return (image.width, image.height)


PromptItem = Union[Text, Tokens, Image]


@dataclass
class Prompt:
    """
    Examples:
        >>> prompt = Prompt.from_text("Provide a short description of AI:")
        >>> prompt = Prompt([
                Image.from_url(url),
                Text.from_text("Provide a short description of AI:"),
            ])
    """

    items: Sequence[PromptItem]

    def __init__(self, items: Union[str, Sequence[PromptItem]]):
        if isinstance(items, str):
            items = [Text(items, [])]
        self.items = items

    @staticmethod
    def from_text(
        text: str, controls: Optional[Sequence[TextControl]] = None
    ) -> "Prompt":
        return Prompt([Text(text, controls or [])])

    @staticmethod
    def from_image(image: Image) -> "Prompt":
        return Prompt([image])

    @staticmethod
    def from_tokens(
        tokens: Sequence[int], controls: Optional[Sequence[TokenControl]] = None
    ) -> "Prompt":
        """
        Examples:
            >>> prompt = Prompt.from_tokens([1, 2, 3])
        """
        return Prompt([Tokens(tokens, controls or [])])

    def to_json(self) -> Sequence[Mapping[str, Any]]:
        return [_to_json(item) for item in self.items]

    @staticmethod
    def from_json(items_json: Sequence[Mapping[str, Any]]) -> "Prompt":
        return Prompt(
            [
                item
                for item in (_prompt_item_from_json(item) for item in items_json)
                if item
            ]
        )


def _prompt_item_from_json(item: Mapping[str, Any]) -> Optional[PromptItem]:
    item_type = item.get("type")
    if item_type == "text":
        return Text.from_json(item)
    if item_type == "image":
        return Image.from_json(item)
    if item_type == "token_ids":
        return Tokens.from_json(item)
    # Skip item instead of raising an error to prevent failures of old clients
    # when item types are extended
    return None


def _to_json(item: PromptItem) -> Mapping[str, Any]:
    if hasattr(item, "to_json"):
        return item.to_json()  # type: ignore
    # Required for backwards compatibility
    # item could be a plain piece of text or a plain list of token-ids
    elif isinstance(item, str):
        return {"type": "text", "data": item}
    elif isinstance(item, List):
        return {"type": "token_ids", "data": item}
    else:
        raise ValueError(
            "The item in the prompt is not valid. Try either a string or an Image."
        )

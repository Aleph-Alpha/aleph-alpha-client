import base64
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Union

from aleph_alpha_client.steering import SteeringConceptCreationResponse
from aleph_alpha_client.structured_output import ResponseFormat
from PIL.Image import Image


class Role(str, Enum):
    """A role used for a message in a chat."""

    User = "user"
    Assistant = "assistant"
    System = "system"


@dataclass(frozen=True)
class Message:
    """
    Describes a message in a chat.

    Parameters:
        role (Role, required):
            The role of the message.

        content (str | List[Union[str | Image]], required):
            The content of the message.
    """

    role: Role
    content: Union[str, List[Union[str, Image]]]

    def to_json(self) -> Mapping[str, Any]:
        result = {
            "role": self.role.value,
            "content": _message_content_to_json(self.content),
        }
        return result


# We introduce a more specific message type because chat responses can only
# contain text at the moment. This enables static type checking to proof that
# `content` is always a string.
@dataclass(frozen=True)
class TextMessage:
    """
    Describes a text message in a chat.

    Parameters:
        role (Role, required):
            The role of the message.

        content (str, required):
            The content of the message.
    """

    role: Role
    content: str

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "TextMessage":
        return TextMessage(
            role=Role(json["role"]),
            content=json["content"],
        )


    # In multi-turn conversations the returned TextMessage is part of the chat
    # history and converted to the prompt. As such, it requires conversion to
    # json again. Here, the message content is a string, but can reuse the 
    # _message_content_to_json function nonetheless.
    def to_json(self) -> Mapping[str, Any]:
        result = {
            "role": self.role.value,
            "content": _message_content_to_json(self.content),
        }
        return result

def _message_content_to_json(content: Union[str, List[Union[str, Image]]]) -> Union[str, List[Mapping[str, Any]]]:
    if isinstance(content, str):
        return content
    else:
        result: List[Mapping[str, Any]] = []
        for chunk in content:
            if isinstance(chunk, str):
                result.append({"type": "text", "text": chunk})
            elif isinstance(chunk, Image):
                result.append({
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(chunk)},
                })
            else:
                raise ValueError(
                    "The item in the prompt is not valid. Try either a string or an Image."
                )
        return result


def _image_to_data_uri(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base_64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base_64}"


@dataclass(frozen=True)
class StreamOptions:
    """
    Additional options to affect the streaming behavior.
    """

    # If set, an additional chunk will be streamed before the data: [DONE] message.
    # The usage field on this chunk shows the token usage statistics for the entire
    # request, and the choices field will always be an empty array.
    include_usage: bool


@dataclass(frozen=True)
class ChatRequest:
    """
    Describes a chat request.

    Only supports a subset of the parameters of `CompletionRequest` for simplicity.
    See `CompletionRequest` for documentation on the parameters.
    """

    model: str
    messages: List[Union[Message, TextMessage]]
    maximum_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stream_options: Optional[StreamOptions] = None
    steering_concepts: Optional[List[str]] = None
    response_format: Optional[ResponseFormat] = None

    def to_json(self) -> Mapping[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        payload["messages"] = [message.to_json() for message in self.messages]
        if self.response_format:
            payload["response_format"] = self.response_format.to_json()
        return payload


class FinishReason(str, Enum):
    """
    The reason the model stopped generating tokens.

    This will be stop if the model hit a natural stop point or a provided stop
    sequence or length if the maximum number of tokens specified in the request
    was reached. If the API is unable to understand the stop reason emitted by
    one of the workers, content_filter is returned.
    """

    Stop = "stop"
    Length = "length"
    ContentFilter = "content_filter"


@dataclass(frozen=True)
class ChatResponse:
    """
    A simplified version of the chat response.

    As the `ChatRequest` does not support the `n` parameter (allowing for multiple return values),
    the `ChatResponse` assumes there to be only one choice.
    """

    finish_reason: FinishReason
    message: TextMessage

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "ChatResponse":
        first_choice = json["choices"][0]
        return ChatResponse(
            finish_reason=FinishReason(first_choice["finish_reason"]),
            message=TextMessage.from_json(first_choice["message"]),
        )


@dataclass(frozen=True)
class Usage:
    """
    Usage statistics for the completion request.

    When streaming is enabled, this field will be null by default.
    To include an additional usage-only message in the response stream, set stream_options.include_usage to true.
    """

    # Number of tokens in the generated completion.
    completion_tokens: int

    # Number of tokens in the prompt.
    prompt_tokens: int

    # Total number of tokens used in the request (prompt + completion).
    total_tokens: int

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "Usage":
        return Usage(
            completion_tokens=json["completion_tokens"],
            prompt_tokens=json["prompt_tokens"],
            total_tokens=json["total_tokens"],
        )


@dataclass(frozen=True)
class ChatStreamChunk:
    """
    A streamed chat completion chunk.

    Parameters:
        content (str, required):
            The content of the current chat completion. Will be empty for the first chunk of every completion stream and non-empty for the remaining chunks.

        role (Role, optional):
            The role of the current chat completion. Will be assistant for the first chunk of every completion stream and missing for the remaining chunks.
    """

    content: str
    role: Optional[Role]

    @staticmethod
    def from_json(choice: Dict[str, Any]) -> "ChatStreamChunk":
        delta = choice["delta"]
        return ChatStreamChunk(
            content=delta["content"],
            role=Role(delta.get("role")) if delta.get("role") else None,
        )


def stream_chat_item_from_json(
    json: Dict[str, Any],
) -> Union[ChatStreamChunk, FinishReason, Usage]:
    """Parse a chat event into one of the three possible types.

    This function takes two assumptions:

    1. If neither a finish reason nor a usage is present, the chunk contains a message.
    2. The chunk only carries information relevant to one of the three possible types, so no information is lost by choosing any one of them.
    """
    if (usage := json.get("usage")) is not None:
        return Usage.from_json(usage)

    first_choice = json["choices"][0]
    if (finish_reason := first_choice.get("finish_reason")) is not None:
        return FinishReason(finish_reason)

    return ChatStreamChunk.from_json(first_choice)

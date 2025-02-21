from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union


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

        content (str, required):
            The content of the message.
    """

    role: Role
    content: str

    def to_json(self) -> Mapping[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "Message":
        return Message(
            role=Role(json["role"]),
            content=json["content"],
        )


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
    messages: List[Message]
    maximum_tokens: Optional[int] = None
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    stream_options: Optional[StreamOptions] = None
    steering_concepts: Optional[List[str]] = None

    def to_json(self) -> Mapping[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        payload["messages"] = [message.to_json() for message in self.messages]
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
    message: Message

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "ChatResponse":
        first_choice = json["choices"][0]
        return ChatResponse(
            finish_reason=FinishReason(first_choice["finish_reason"]),
            message=Message.from_json(first_choice["message"]),
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

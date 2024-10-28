from dataclasses import dataclass, asdict
from typing import List, Optional, Mapping, Any, Dict, Union
from enum import Enum


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

    def to_json(self) -> Mapping[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        payload["messages"] = [message.to_json() for message in self.messages]
        return payload


@dataclass(frozen=True)
class ChatResponse:
    """
    A simplified version of the chat response.

    As the `ChatRequest` does not support the `n` parameter (allowing for multiple return values),
    the `ChatResponse` assumes there to be only one choice.
    """
    finish_reason: str
    message: Message

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "ChatResponse":
        first_choice = json["choices"][0]
        return ChatResponse(
            finish_reason=first_choice["finish_reason"],
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
            total_tokens=json["total_tokens"]
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
    def from_json(json: Dict[str, Any]) -> Optional["ChatStreamChunk"]:
        """
        Returns a ChatStreamChunk if the chunk contains a message, otherwise None.
        """
        if not (delta := json["choices"][0]["delta"]):
            return None

        return ChatStreamChunk(
            content=delta["content"],
            role=Role(delta.get("role")) if delta.get("role") else None,
        )


def stream_chat_item_from_json(json: Dict[str, Any]) ->  Union[Usage, ChatStreamChunk, None]:
    if (usage := json.get("usage")) is not None:
        return Usage.from_json(usage)

    return ChatStreamChunk.from_json(json)
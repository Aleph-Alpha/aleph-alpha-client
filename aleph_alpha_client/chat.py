from dataclasses import dataclass, asdict
from typing import List, Optional, Mapping, Any, Dict
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
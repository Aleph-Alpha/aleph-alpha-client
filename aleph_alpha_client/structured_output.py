from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union


@dataclass(frozen=True)
class JSONSchema:
    """
    JSON schema that structured output must adhere to.

    Parameters:
        json_schema:
            JSON schema that structured output must adhere to.

    Examples:
        >>> schema = JSONSchema(
        >>>     schema={'type': 'object', 'properties': {'bar': {'type': 'integer'}}},
        >>>     name="example_schema",
        >>>     description="Example schema with a bar integer property",
        >>>     strict=True
        >>> )
    """

    schema: Mapping[str, Any]
    name: str
    description: Optional[str] = None
    strict: Optional[bool] = False

    def to_json(self) -> Mapping[str, Any]:
        return {"type": "json_schema", "json_schema": asdict(self)}


ResponseFormat = Union[JSONSchema]

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union
from pydantic import BaseModel


@dataclass(frozen=True)
class JSONSchema:
    """
    JSON schema that structured output must adhere to.

    Parameters:
        schema:
            JSON schema that structured output must adhere to.
        name:
            Name of the schema.
        description:
            Description of the schema.
        strict:
            Whether the schema should be strictly enforced.

    Examples:
        >>> schema = JSONSchema(
        >>>     schema={
        >>>         'type': 'object',
        >>>         'title': 'Aquarium',
        >>>         'properties': {
        >>>             'nemo': {
        >>>                 'type': 'string',
        >>>                 'title': 'Nemo'
        >>>             }
        >>>         },
        >>>         'required': ['nemo']
        >>>     },
        >>>     name="aquarium",
        >>>     description="Describe nemo",
        >>>     strict=True
        >>> )
    """

    schema: Mapping[str, Any]
    name: str
    description: Optional[str] = None
    strict: Optional[bool] = False

    def to_json(self) -> Mapping[str, Any]:
        return {"type": "json_schema", "json_schema": asdict(self)}

    @classmethod
    def from_pydantic(cls, model_class) -> "JSONSchema":
        """
        Create a JSONSchema from a Pydantic model class.
        
        Parameters:
            model_class: A Pydantic BaseModel class
            
        Returns:
            JSONSchema instance with the schema generated from the Pydantic model
            
        Raises:
            ValueError: If the provided class is not a Pydantic BaseModel
        """
        # Validate that the model_class is a Pydantic BaseModel
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise ValueError(f"Expected a Pydantic BaseModel class, got {type(model_class).__name__}")
        
        schema = model_class.model_json_schema()
        name = getattr(model_class, '__name__', 'generated_schema')

        # Use the name as default if no description is provided
        description = getattr(model_class, '__doc__', name)

        return cls(
            schema=schema,
            name=name,
            description=description,
            strict=True
        )


# Define ResponseFormat type - use Any to support both JSONSchema and Pydantic models
ResponseFormat = Union[JSONSchema, Any]

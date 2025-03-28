import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, NewType, Union
import pytest
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.prompt import Image, Prompt


@pytest.fixture(scope="session")
def sync_client() -> Client:
    return Client(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        # This will retry after [0.0, 0.25, 0.5, 1.0, 2.0] seconds
        total_retries=int(os.environ.get("TEST_API_RETRIES", "5")),
    )


@pytest.fixture()
async def async_client() -> AsyncIterable[AsyncClient]:
    async with AsyncClient(
        token=get_env_var("TEST_TOKEN"),
        host=get_env_var("TEST_API_URL"),
        # This will retry after [0.0, 0.25, 0.5, 1.0, 2.0] seconds
        total_retries=int(os.environ.get("TEST_API_RETRIES", "5")),
    ) as client:
        yield client


@pytest.fixture(scope="session")
def model_name() -> str:
    return "luminous-base"


@pytest.fixture(scope="session")
def chat_model_name() -> str:
    return "llama-3.1-8b-instruct"


@pytest.fixture(scope="session")
def prompt_image() -> Image:
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    return Image.from_file(image_source_path)


def get_env_var(env_var: str) -> str:
    value = os.environ.get(env_var)
    if value is None:
        raise ValueError(
            f"Test parameters could not be read from .env. Make sure to create a .env file with the key {env_var}"
        )
    return value


@dataclass(frozen=True)
class PhariaAiFeatureSet:
    _Stable = NewType("_Stable", int)

    class _Special(enum.Enum):
        BETA = 1

    _value: Union[_Stable, _Special]

    @classmethod
    def stable(cls, version: int):
        return PhariaAiFeatureSet(cls._Stable(version))

    @classmethod
    def beta(cls):
        return PhariaAiFeatureSet(cls._Special.BETA)

    @classmethod
    def from_env(cls) -> "PhariaAiFeatureSet":
        env_name = "PHARIA_AI_FEATURE_SET"
        env_value = os.environ.get(env_name)
        if env_value is not None:
            value = env_value.strip()
            if value.lower() == "beta":
                return cls.beta()
            elif value.isdecimal():
                return cls.stable(int(value))
            else:
                raise ValueError(
                    f"environment variable {env_name} is invalid: {value}, "
                    "only integers or 'BETA' are supported"
                )
        else:
            return cls.stable(1)

    def __lt__(self, other: "PhariaAiFeatureSet") -> bool:
        if isinstance(self._value, int):
            if isinstance(other._value, int):
                return self._value < other._value
            else:  # other is _Special
                return True
        else:  # self is _Special
            if isinstance(other._value, self._Special):
                return self._value.value < other._value.value
            else:  # other is _Stable
                return False


# Tests with this decorator only run if PHARIA_AI_FEATURE_SET is set to "BETA".
requires_beta_features = pytest.mark.skipif(
    PhariaAiFeatureSet.from_env() < PhariaAiFeatureSet.beta(),
    reason="requires beta features",
)


def llama_prompt(text: str) -> Prompt:
    return Prompt.from_text(
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

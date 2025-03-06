import importlib.metadata
from typing import Dict

MIN_API_VERSION = "1.19.0"


__version__ = importlib.metadata.version("aleph-alpha-client")


def user_agent_headers() -> Dict[str, str]:
    """User agent that should be send for specific versions of the SDK."""

    return {"User-Agent": "Aleph-Alpha-Python-Client-" + __version__}

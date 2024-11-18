import importlib.metadata
from typing import Dict
import re
from pathlib import Path
import logging


MIN_API_VERSION = "1.19.0"


def pyproject_version() -> str:
    """Return the package version specified in pyproject.toml.

    Poetry inject the package version from the pyproject.toml file into the package
    metadata at build time. This means any user of the package has access to the
    version (it is send as User-Agent header in requests).

    If this repository is cloned and the package is not installed, the version will
    be not available (it is 0.0.0 per default). To also provide a version in these
    cases, this function tries to read the version from the pyproject.toml file.

    To not break imports in cases where both, the pyproject.toml file and the package
    metadata are not available, no error is raised and a default version of 0.0.0
    will be returned. One such case is building the package with `pip install git+url`,
    where pip is not able to read the [tool.poetry.version] field and also does not
    keep the pyproject.toml file when the package is installed.
    """
    NO_VERSION = "0.0.0"
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"

    if not pyproject_path.is_file():
        logging.error("pyproject.toml file not found.")
        return NO_VERSION

    version_pattern = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)

    with pyproject_path.open("r", encoding="utf-8") as file:
        content = file.read()

    if match := version_pattern.search(content):
        return match.group(1)

    logging.error("Version not found in pyproject.toml")
    return NO_VERSION


__version__ = importlib.metadata.version("aleph-alpha-client")

if __version__ == "0.0.0":
    __version__ = pyproject_version()


def user_agent_headers() -> Dict[str, str]:
    """User agent that should be send for specific versions of the SDK.

    For some installations, the package version is not available (== 0.0.0).
    Setting the user agent header to "Aleph-Alpha-Python-Client-0.0.0" causes the
    API to return a response with some fields omitted (due to a bug with older
    clients which can not handle the new fields). These omitted fields in turn cause
    new clients to fail on deserialization. To prevent these errors, we omit the
    user agent header in cases where the version is not available (== 0.0.0).
    """
    if __version__ == "0.0.0":
        return {}
    else:
        return {"User-Agent": "Aleph-Alpha-Python-Client-" + __version__}

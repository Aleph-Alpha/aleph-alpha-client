from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


def version():
    exec(open("aleph_alpha_client/version.py").read())
    return locals()["__version__"]


tests_require = [
    "pytest",
    "pytest-cov",
    "pytest-dotenv",
    "pytest-httpserver",
    "pytest-aiohttp",
]

types_require = ["mypy", "types-requests"]

dev_require = (
    tests_require
    + types_require
    + [
        "nbconvert",
        "ipykernel",
        "black",
    ]
)

docs_require = ["sphinx", "sphinx_rtd_theme"]

setup(
    name="aleph-alpha-client",
    url="https://github.com/Aleph-Alpha/aleph-alpha-client",
    author="Aleph Alpha",
    author_email="support@aleph-alpha.com",
    packages=["aleph_alpha_client"],
    # urllib is used directly for retries
    install_requires=[
        "requests >= 2.28",
        "urllib3 >= 1.26",
        "aiohttp >= 3.8.3",
        "aiodns >= 3.0.0",
        "aiohttp-retry >= 2.8.3",
        "tokenizers >= 0.13.2",
    ],
    tests_require=tests_require,
    extras_require={
        "test": tests_require,
        "types": types_require,
        "dev": dev_require,
        "docs": docs_require,
    },
    license="MIT",
    description="python client to interact with Aleph Alpha api endpoints",
    long_description=readme(),
    long_description_content_type="text/markdown",
    version=version(),
)

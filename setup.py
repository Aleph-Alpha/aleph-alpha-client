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

setup(
    name="aleph-alpha-client",
    url="https://github.com/Aleph-Alpha/aleph-alpha-client",
    author="Aleph Alpha",
    author_email="support@aleph-alpha.com",
    packages=["aleph_alpha_client"],
    install_requires=["requests"],
    tests_require=tests_require,
    extras_require={"test": tests_require, "types": types_require, "dev": dev_require},
    license="MIT",
    description="python client to interact with Aleph Alpha api endpoints",
    long_description=readme(),
    long_description_content_type="text/markdown",
    version=version(),
)

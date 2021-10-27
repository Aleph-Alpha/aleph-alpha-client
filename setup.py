from setuptools import setup

setup(
    name="aleph_alpha_client",
    url="https://github.com/Aleph-Alpha/aleph-alpha-client",
    author="Aleph Alpha",
    author_email="info@aleph-alpha.de",
    packages=["aleph_alpha_client"],
    install_requires=["requests"],
    tests_require=["pytest", "pytest-cov", "python-dotenv"],
    extras_require={
        "test": ["pytest", "pytest-cov", "python-dotenv"],
    },
    version="1.0.0",
    license="MIT",
    description="python client to interact with Aleph Alpha api endpoints",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)

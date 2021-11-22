from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="aleph-alpha-client",
    url="https://github.com/Aleph-Alpha/aleph-alpha-client",
    author="Aleph Alpha",
    author_email="support@aleph-alpha.de",
    packages=["aleph_alpha_client"],
    setup_requires=['setuptools_scm'],
    install_requires=["requests"],
    tests_require=["pytest", "pytest-cov", "python-dotenv"],
    extras_require={
        "test": ["pytest", "pytest-cov", "python-dotenv"],
    },
    use_scm_version=True,
    license="MIT",
    description="python client to interact with Aleph Alpha api endpoints",
    long_description=readme(),
    long_description_content_type="text/markdown",
)

from setuptools import setup

setup(
    name='AlephAlphaClient',
    url='https://github.com/Aleph-Alpha/aleph-alpha-client',
    author='Aleph Alpha',
    author_email='info@aleph-alpha.de',
    packages=['AlephAlphaClient'],
    install_requires=['requests'],
    tests_require=['pytest', 'pytest-cov', 'python-dotenv'],
    version='0.0.1',
    license='MIT',
    description='python client to interact with Aleph Alpha api endpoints',
    long_description=open('README.md').read(),
)

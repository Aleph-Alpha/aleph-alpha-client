# Contributions

## Testing

Tests use pytests with (optional) coverage plugin. Install the locally cloned repo in editable mode with:

```bash
pip install -e .[test]
```

**Tests make api calls that reduce your quota!**

### Run tests

Tests can be run using pytest. Make sure to create a `.env` file with the following content:

```env
# test settings
TEST_API_URL=https://test.api.aleph-alpha.com
TEST_TOKEN=your_token
```

Instead of a token username and password can be used.

```env
# test settings
TEST_API_URL=https://api.aleph-alpha.com
TEST_USERNAME=your_username
TEST_PASSWORD=your_password
```

- A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)
- A subset of tests can be selected by pointing to the module within tests

```bash
# run all tests, output coverage report of aleph_alpha_client module in terminal
pytest --cov-report term --cov=aleph_alpha_client tests
pytest tests -v # start verbose
```

If an html coverage report has been created a simple http server can be run to serve static files.

```bash
python -m http.server --directory htmlcov 8000
```

## Conventional Commits

Please use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages. This allows [release-please](https://github.com/googleapis/release-please) to update the Changelog and version number according to Semantic Versioning.

## Releasing a new version

1. Merge your changes to main
2. A PR will automatically be created by [release-please](https://github.com/googleapis/release-please)
3. Merge the PR. This will update the version number and Changelog of the package, trigger a GitHub release and push the changes to PyPi.

## Working on our documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) for our documentation and publish it on [Read the Docs](https://aleph-alpha-client.readthedocs.io/en/latest/).
To work on the documentation, you need to install the project editable and with the `docs` extra.

```bash
pip install -e .[docs]
```

The documentation can be generated with:

```bash
cd docs
make html
```

Make sure that the documentation can be generated without Sphinx warnings or errors.

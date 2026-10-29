# Contributing

Open an issue with the command, configuration, dependency versions, and a minimal
synthetic example. Do not attach patient images, identifiers, demographic tables,
private paths, access tokens, or unpublished model weights.

For code changes, install `pip install -e '.[dev]'` and run `pytest -q`. Changes to
losses, subgroup definitions, preprocessing or evaluation should include an
analytical/synthetic check and an explanation of whether they change the paper
protocol. Keep scientific results separate from software smoke-test output.

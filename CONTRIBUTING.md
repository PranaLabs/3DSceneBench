# Contributing

Thank you for looking at 3DSceneBench. This page says where to ask, what to send and how to check a change.

## Questions and bugs

Open an issue. For a bug, say what you ran, what you expected and what happened, and attach the smallest scene that shows it. A `reward.json` carries `verifier_version`; give that too.

**A way to score points without building the world** (a scoring loophole) is a security matter for a benchmark. Please do not open a public issue for it: see [SECURITY.md](SECURITY.md).

## Changes

1. Fork the repository and make your change on a branch.
2. Keep one pull request to one subject.
3. Run the formatter and the self-tests (below).
4. Say in the pull request what changes for an entrant's score, if anything. A change that moves scores needs a maintainer's decision and a new verifier version, because results from different versions cannot be compared.

## Checking a change

The code is formatted with `ruff format` at the width in `ruff.toml`:

```bash
ruff format .
```

The self-tests run with Python's `unittest` in an environment with `requirements/core.txt`:

```bash
python -m unittest discover -s tests -t .
```

Some self-tests need Blender, Genesis or the image-metric weights and skip themselves without them. `requirements/` and `docker/` hold the full environment.

## Words

One check of a world is a **test**, as in the paper: a push, a tilt, a pull on a drawer. The programs that run the tests and score a world are the **verifiers**. Our own unit tests of this code are **self-tests**. What an agent hands in is a **world**, and the room it is compared with is the **reference room**. Please keep to these words.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, as the rest of the repository ([LICENSE](LICENSE)).

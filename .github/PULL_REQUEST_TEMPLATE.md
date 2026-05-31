<!--
Thanks for the PR!  Please fill in the sections below.  Trim anything that
doesn't apply.
-->

## What does this PR do?

<!-- 1–3 sentences.  What's the user-visible change? -->

## Why?

<!-- Linked issue, motivation, design decision. -->

Closes #

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue) → PATCH
- [ ] New feature (non-breaking change which adds functionality) → MINOR
- [ ] New backend / model integration → MINOR
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected) → MAJOR
- [ ] Docs / tooling only → no release
- [ ] Refactor (no functional change) → no release

## Checklist

- [ ] Pre-commit ran clean (`pre-commit run --all-files`)
- [ ] Tests pass (`pytest -s mlx_vc/tests/ -v`)
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/) format
- [ ] If this adds a new backend: added an entry in `BACKENDS`, included a setup recipe in the script docstring, and verified it runs end-to-end
- [ ] If this adds a public API: docstring + at least one unit test
- [ ] CHANGELOG.md updated under `[Unreleased]` if this is user-visible
- [ ] If model quality changed: ran `scripts/evaluate_quality.py` and added results to BENCHMARK.md Part B

## Testing notes

<!-- Anything reviewers should run / look at / be aware of? -->

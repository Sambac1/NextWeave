# Contributing to NextWeave

## Development setup

Follow the installation steps in [README.md](README.md). Use a local backend data directory and configure your own model credentials. Tests should use fixtures and mocked model responses whenever possible.

## Change guidelines

- Keep frontend presentation separate from backend evidence and simulation logic.
- Preserve source provenance, evidence cutoffs, actor visibility boundaries, and probability semantics.
- Keep code comments in English. Explain non-obvious constraints or invariants; avoid narrating the implementation or recording development history.
- Keep documentation focused on current behavior and reproducible setup.
- Do not commit credentials, local environment files, logs, runtime analyses, caches, or build output.
- Preserve the fixed comparison archive independently of working evaluation data.

## Validation

Run `npm run check` and `npm run build` at the project root. Run `python -m pytest -q` from `backend/` with its virtual environment activated. For a behavior change, add a focused regression test that exercises the public behavior rather than reproducing the implementation.

A change description should state the problem, the resulting behavior, and the checks performed. Include screenshots for changes that affect the interface. Describe any model-backed check separately from offline tests, including its inputs and resource limits.

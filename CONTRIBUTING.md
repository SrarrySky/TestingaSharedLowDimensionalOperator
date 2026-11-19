# Contributing

Thank you for contributing to DLC-SSM Risk Geometry.

## Development workflow

1. Create a focused branch.
2. Install `pip install -e ".[dev]"`.
3. Run `ruff check .`.
4. Run `ruff format --check src/cli.py tests/test_release_tooling.py`.
5. Run `pytest -q`.
6. Add or update tests for all behavioral changes.
7. Do not commit Tushare tokens, raw licensed data, model outputs containing restricted data, or future-information leakage.

## Scientific changes

A pull request that changes an estimand, causal-support interpretation, split, loss function, stability rule, or uncertainty treatment must update:

- the matching YAML configuration;
- `docs/METHODS.md`;
- at least one test;
- the reproducibility notes.

Do not weaken the train/validation/test boundaries to improve reported results.

## Documentation changes

The operational experiment contract is documented in:

- `docs/METHODS.md` for mathematical, algorithmic, API, extension, and experiment contracts;
- `docs/PAPER_TO_CODE.md` for paper-to-code-to-evidence traceability;
- `docs/REPRODUCIBILITY.md` for command-line entry points, release boundaries, result-audit procedures, and troubleshooting;
- `paper_supplement/MATHEMATICAL_DERIVATIONS.md` for complete derivations of the paper's propositions, corollary, estimation equations, and diagnostics;
- `paper_supplement/ROBUSTNESS_AND_UNCERTAINTY_RESULTS.md` for the post-acceptance robustness and uncertainty evidence record.

Security disclosures follow `SECURITY.md`; contribution conduct follows
`CODE_OF_CONDUCT.md`.

Changes to an estimand, split, loss, support method, uncertainty treatment or selection rule must update both the relevant configuration and these documents.

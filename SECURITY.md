# Security policy

## Scope

This is a research-code repository. Do not commit Tushare tokens, credentials, private data, licensed raw data, generated local outputs, or manuscript working files. The supported public-source boundary is documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reporting a vulnerability or accidental disclosure

Do not open a public issue containing a credential, private dataset, or other sensitive material. Contact the repository maintainers privately through the contact information associated with the GitHub repository and include:

- the affected commit or path;
- the type of disclosure or vulnerability;
- steps to reproduce, if applicable; and
- whether a credential has already been revoked.

Immediately revoke and rotate any exposed token. Removing a file in a later commit does not remove it from Git history; historical cleanup must be handled as a separate release operation.

## Dependency and release hygiene

The CI workflow runs the test suite and release checks. Before publishing a release, inspect tracked files and history for secrets, local absolute paths, licensed data, generated outputs, and manuscript-only material.

# Changelog

## 0.2.1rc5 — 2026-10-06

- Use `dose-combination-bo` for the software and command, and `dose_combination_bo` for Python imports.
- Update installation files, examples, tests and manuscript references together.
- Keep the numerical algorithms and saved research results unchanged.

## Documentation revision — 2026-10-06

- Link the current article, supporting information and examples from the README.
- Clarify installation from a downloaded or cloned repository.
- Remove review-package links and update the reading guide and release status.
- Update links for the repository name `dose-combination-bo`; the then-current software name was unchanged.
- Keep software version 0.2.1rc4; algorithms and reported simulation results are unchanged.

## Documentation revision — 2026-09-08

- Shorten the README and explain how to install the package and read its output.
- Combine scope and reporting definitions; remove unimplemented extension plans.
- Preserve the historical analysis specification and all result tables.
- Rebuild distributions with the current documentation. Package version remains 0.2.1rc4; Python modules, tests and examples are unchanged.

## 0.2.1rc4 — 2026-09-07

- Stabilize probabilities and log probabilities for narrow normal intervals using a bounded local-density expansion and scaled tails.
- Keep log probabilities finite when ordinary probabilities underflow; retain negative cKG values.
- Add ten regression tests and sixty high-precision integral cases.

## 0.2.1rc3 — 2026-09-07

- Restore isolated builds; add regression tests and real GP/Ray checks.
- Test the installed wheel in CI and compare 31 matched runs with the study code.

## 0.2.1rc2 — integrated v12 baseline

- Check inputs and callback scores; enforce assignment restrictions and select only opened doses in short-budget trials.
- Record the version and source in checkpoints; distinguish distance to the grid optimum from the earlier distance metric.

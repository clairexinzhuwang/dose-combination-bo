# Software verification — 6 October 2026

Software: **dose-combination-bo 0.2.1rc5**. Python import: `dose_combination_bo`.
Command: `dose-combination-bo`.

The renamed wheel was installed after removing the previous package from the
project's isolated Python 3.12.7 environment. System packages are disabled.
Tests and examples ran outside the source checkout. Every installed Python
module matched the maintained source, and dependency checks passed.

| Suite | Passed | Failed | Import errors | Other errors | Skipped | Excluded | Not executed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maintained regression, including real GP fitting and two Ray tests | 237 | 0 | 0 | 0 | 0 | 0 | 0 |
| Numerical and controller checks with explicit posterior fixtures | 148 | 0 | 0 | 0 | 0 | 0 | 0 |

The second suite checks algebra and controller behavior; it does not establish
GP integration. Both suites ran without marker filters. README CLI and Python
examples, both extension scripts, and both CLI plugin paths passed. The narrow
normal-interval counterexample remains covered by the regression suite.

Environment: macOS arm64; NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.10.0,
GPyTorch 1.15.2, Ray 2.54.1, pytest 7.4.4 and mpmath 1.3.0. Library warnings
were reported by pytest; no failures were suppressed.

The name, import path, command and version changed together. The 20 current
Python modules differ from the preceding candidate only by naming and version
substitutions. A new installed-wheel run on the same fixed 4-, 20- and 40-record
datasets reproduced the original observations, fitted posteriors and all score
vectors exactly. New timing records are separate; the paper's original timing
evidence is identified as pre-rename candidate 0.2.1rc4 and was not rewritten.

Saved study results, original source contracts and the frozen controller
reference are unchanged. Their original identifiers remain historical evidence.
This rename does not rerun the paper's adaptive trials or replace the earlier
31-pair behavior comparison. The complete calibration path archive is still
separate; see [Reproduction](../docs/reproducibility.md).

[GitHub Actions](https://github.com/clairexinzhuwang/dose-combination-bo/actions)
runs fresh isolated installations, the two suites and examples, retaining its
own reports. This document records the local acceptance run. Detailed logs,
JUnit files and the rename comparison are retained outside the public candidate.

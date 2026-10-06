# Software verification

Software: **dose-combination-bo 0.2.1rc5**. Checked **6 October 2026**.

The renamed wheel passed **237 maintained tests**, including real GP fitting
and two Ray tests, plus **148 numerical/controller checks** using explicit
posterior fixtures. Both suites had zero failures, import errors, other errors,
skips, exclusions and unexecuted cases. The README and guide examples and both
CLI plugin paths passed outside the source checkout.

The fixed-data check also reproduced the earlier observations, fitted
predictions and all four rules' score vectors exactly. The numerical algorithms
and saved research results are unchanged.

The [verification report](distributions/VERIFICATION_REPORT.md) records the
package, environment and scope. [GitHub Actions](https://github.com/clairexinzhuwang/dose-combination-bo/actions)
provides separate hosted checks. [Reproduction](docs/reproducibility.md)
explains the included records and the separate complete research archive.

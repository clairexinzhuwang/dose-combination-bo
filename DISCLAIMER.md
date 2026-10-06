# Research use

This software simulates two-agent dose-finding studies with continuous efficacy
and toxicity outcomes. It has not been validated for clinical trial operation.

The toxicity rule concerns mean toxicity. If no combination passes the rule,
the simulator can make a fallback selection. A selected combination may never
have been assigned. Counts of assignments above the mean-toxicity limit do not
count observed adverse events.

See [Scope and limitations](docs/scope_and_limitations.md) for the evaluated
settings. The paper and data have separate terms from the software's MIT license.

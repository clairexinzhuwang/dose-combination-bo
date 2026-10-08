# Bayesian Optimization for Combination Dose Finding

`dose-combination-bo` compares four ways to choose doses in a simulated trial. It reports
which combinations participants receive and which combination is chosen at the
end. The current software and paper use combinations of two agents.

## Paper

The companion paper is available as
[arXiv:2610.09245](https://arxiv.org/abs/2610.09245)
([PDF](https://arxiv.org/pdf/2610.09245)):

> Xinzhu Wang and Tanzy Love. *Bayesian Optimization for Dose Finding with Two
> Agents: Participant Allocation and Final Selection*. arXiv:2610.09245
> [stat.AP], 2026.

Versioned repository copies of the [article](paper/Main_Manuscript.pdf) and
[supporting information](paper/Supporting_Information.pdf) are also included,
along with [both in one PDF](paper/Complete_Manuscript.pdf).

## Install and run

Python 3.10 or later is required; Python 3.12 is the tested configuration.
Clone the repository and install it in a virtual environment:

```bash
git clone https://github.com/clairexinzhuwang/dose-combination-bo.git
cd dose-combination-bo
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt ".[tables]"
dose-combination-bo list
dose-combination-bo run --acquisitions cKG,cEI,tmse,entropy --surfaces osa --seeds 2 --budget 12 --serial
```

If you downloaded a ZIP instead, open the extracted directory and start with
the `python -m venv .venv` step.
On Windows, activate with `.venv\Scripts\activate` instead. The last command
runs a small demonstration. Increase the budget and replicate count for a
simulation study. The [wheel and source distribution](distributions/) are also
available.

For one trial in Python:

```python
import dose_combination_bo as dcb

result = dcb.run_trial("cKG", seed=0, z=0, gamma=0.7, sim="osa", budget=12)
print(result["selection_status"])
print(result["rec_true_eff"], result["rec_true_tox"])
```

## Using the software

The [software guide](GETTING_STARTED.md) explains the four rules, settings,
output, saved results and tests. It also shows how to add your own rule or
scenario.

The toxicity limit applies to a combination's mean response; it does not give
an individual participant's risk of an adverse event. The software is for
simulation studies. A clinical trial needs its own validated protocol.

Software license: [MIT](LICENSE). Citation details are in [CITATION.cff](CITATION.cff).
The [manuscript terms](paper/LICENSE.md) are separate from the software license.
The companion paper is *Bayesian Optimization for Dose Finding with Two Agents:
Participant Allocation and Final Selection*, by Xinzhu Wang and Tanzy Love.

# Migrated from study/tests/test_oc.py; see REGRESSION_MIGRATION.csv.
"""Hand-checked tests for the post-processing operating-characteristic reports."""
import builtins
import json
from pathlib import Path

import numpy as np
import pytest

import dose_combination_bo as sdb


def _ensure_handcheck_surface():
    try:
        return sdb.get_surface("oc_handcheck")
    except KeyError:
        @sdb.surface("oc_handcheck")
        def handcheck(z):
            eff = lambda a, b: float(a + b)
            tox = lambda a, b: float(a + b)
            return dict(gd=1.0, dopt=(0.5, 0.5), fopt=1.0,
                        eff=eff, tox=tox, sf=0.2, sg=0.2)
        return handcheck


def _handchecked_results():
    spec = _ensure_handcheck_surface()(0)
    recommendations = {
        "cEI": [(0.5, 0.5), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0)],
        "random": [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)],
    }
    toxic = [0, 4, 8, 10]
    gate_size = [2, 0, 4, 1]
    gate_safe = [1, 0, 3, 1]
    rows = []
    for policy, doses in recommendations.items():
        for seed, dose in enumerate(doses):
            rec_tox = spec["tox"](*dose)
            rows.append({
                "policy": policy,
                "seed": seed,
                "sim": "oc_handcheck",
                "stratum": 0,
                "gamma": 0.7,
                "mode": "latent",
                "noise": "fixed",
                "kap": 1.0,
                "grid_n": 3,
                "budget": 10,
                "r_k": 2,
                "warmup": 2,
                "empty_gate": "pf",
                "dose_units": float(np.linalg.norm(np.asarray(dose) - spec["dopt"]) / 0.5),
                "rec_d1": dose[0],
                "rec_d2": dose[1],
                "rec_true_eff": spec["eff"](*dose),
                "rec_true_tox": rec_tox,
                "rec_unsafe": int(rec_tox > spec["gd"]),
                "toxic": toxic[seed],
                "n_gate_pass": gate_size[seed],
                "n_gate_pass_safe": gate_safe[seed],
            })
    return rows


def _records(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else table


def _stopped_results():
    """Four hand-checkable program outcomes: two recommendations and two stops."""
    spec = _ensure_handcheck_surface()(0)
    common = {
        "policy": "cEI",
        "sim": "oc_handcheck",
        "stratum": 0,
        "gamma": 0.7,
        "mode": "latent",
        "noise": "fixed",
        "kap": 1.0,
        "grid_n": 3,
        "budget": 10,
        "r_k": 2,
        "warmup": 2,
        "empty_gate": "pf",
        "protocol_scaffold": "start_low_expansion",
        "region_step": 0.25,
        "empty_gate_stop_after": 3,
        "exclude_repeats_during_expansion": True,
        "initialization_size": 2,
    }
    recommended = []
    for seed, (dose, n_enrolled, toxic) in enumerate([
        ((0.5, 0.5), 10, 1),
        ((1.0, 0.5), 10, 3),
    ]):
        rec_tox = spec["tox"](*dose)
        recommended.append(dict(
            common,
            seed=seed,
            recommendation_made=True,
            stop_reason=None,
            n_enrolled=n_enrolled,
            n_unique_doses=5,
            n_empty_gate_events=seed,
            initialization_patients_above=0,
            post_initialization_patients_above=toxic,
            dose_units=float(np.linalg.norm(np.asarray(dose) - spec["dopt"]) / 0.5),
            rpsel=0.25,
            rec_d1=dose[0],
            rec_d2=dose[1],
            rec_true_eff=spec["eff"](*dose),
            rec_true_tox=rec_tox,
            rec_unsafe=int(rec_tox > spec["gd"]),
            toxic=toxic,
            n_gate_pass=2,
            n_gate_pass_safe=1,
        ))

    stopped = []
    for seed, (n_enrolled, toxic, n_unique, empty_events) in enumerate([
        (6, 2, 3, 3),
        (8, 4, 4, 4),
    ], start=2):
        stopped.append(dict(
            common,
            seed=seed,
            recommendation_made=False,
            stop_reason="NO_FEASIBLE_DOSE",
            n_enrolled=n_enrolled,
            n_unique_doses=n_unique,
            n_empty_gate_events=empty_events,
            initialization_patients_above=0,
            post_initialization_patients_above=toxic,
            dose_units=None,
            rpsel=None,
            rec_d1=None,
            rec_d2=None,
            rec_true_eff=None,
            rec_true_tox=None,
            rec_unsafe=None,
            toxic=toxic,
            n_gate_pass=0,
            n_gate_pass_safe=0,
        ))
    return recommended + stopped


def test_oc_table_handchecked_mcse():
    rows = _records(sdb.oc_table(_handchecked_results(), sim="oc_handcheck", gamma=0.7))
    cei = next(row for row in rows if row["policy"] == "cEI")

    assert cei["n"] == 4
    assert cei["n_rows"] == 4
    assert cei["pcs"] == pytest.approx(50.0)
    assert cei["pcs_mcse"] == pytest.approx(
        100 * np.std([1, 1, 0, 0], ddof=1) / 2
    )
    assert cei["pcs_within1"] == pytest.approx(75.0)
    assert cei["pcs_within1_mcse"] == pytest.approx(
        100 * np.std([1, 1, 1, 0], ddof=1) / 2
    )
    assert cei["rec_unsafe_pct"] == pytest.approx(50.0)
    assert cei["rec_efficacy"] == pytest.approx(1.375)
    assert cei["rec_efficacy_mcse"] == pytest.approx(
        np.std([1.0, 1.0, 1.5, 2.0], ddof=1) / 2
    )
    assert cei["efficacy_regret"] == pytest.approx(-0.375)
    assert cei["pct_patients_above"] == pytest.approx(55.0)
    assert cei["pct_patients_above_mcse"] == pytest.approx(
        np.std([0.0, 40.0, 80.0, 100.0], ddof=1) / 2
    )
    assert cei["overdose60"] == pytest.approx(50.0)
    assert cei["overdose80"] == pytest.approx(25.0)  # strict > .80, not >= .80
    assert cei["gate_size"] == pytest.approx(1.75)
    assert cei["gate_empty_pct"] == pytest.approx(25.0)
    assert cei["gate_precision_n"] == 3
    assert cei["gate_precision"] == pytest.approx(0.75)
    assert cei["gate_precision_mcse"] == pytest.approx(
        np.std([0.5, 0.75, 1.0], ddof=1) / np.sqrt(3)
    )


def test_oc_table_always_groups_by_policy_and_labels_random_reference():
    rows = _records(sdb.oc_table(_handchecked_results(), by=("stratum",)))
    assert {(row["policy"], row["stratum"]) for row in rows} == {
        ("cEI", 0), ("random", 0)
    }
    random = next(row for row in rows if row["policy"] == "random")
    assert random["policy_label"] == "random among eligible combinations"
    assert random["chance_control"] is True


def test_selection_table_handchecked_and_sums_to_100():
    table = sdb.selection_table(_handchecked_results(), "oc_handcheck", z=0, gamma=0.7)
    rows = _records(table)
    cei = "cEI_selected_pct"
    random = "random among eligible combinations_selected_pct"
    assert len(rows) == 9
    assert sum(row[cei] for row in rows) == pytest.approx(100.0)
    assert sum(row[random] for row in rows) == pytest.approx(100.0)
    target = next(row for row in rows if row["true_obd"])
    assert (target["d1"], target["d2"]) == (0.5, 0.5)
    assert target[cei] == pytest.approx(50.0)
    assert target[f"{cei}_mcse"] == pytest.approx(
        100 * np.std([1, 1, 0, 0], ddof=1) / 2
    )
    assert target["cEI_n"] == 4
    assert target["cEI_n_rows"] == 4
    assert target["true_eff"] == pytest.approx(1.0)
    assert target["true_tox"] == pytest.approx(1.0)
    assert target["feasible"] is True


def test_oc_table_separates_program_and_recommendation_conditional_estimands():
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        row = _records(sdb.oc_table(_stopped_results()))[0]

    assert row["n"] == 4
    assert row["n_rows"] == 4
    assert row["recommendation_pct"] == pytest.approx(50.0)
    assert row["no_feasible_dose_pct"] == pytest.approx(50.0)
    assert row["pcs"] == pytest.approx(25.0)
    assert row["pcs_within1"] == pytest.approx(50.0)
    assert row["rec_unsafe_pct"] == pytest.approx(25.0)
    assert row["expected_n"] == pytest.approx(8.5)

    assert row["rec_efficacy"] == pytest.approx(1.25)
    assert row["rec_efficacy_n"] == 2
    assert row["rec_efficacy_n_rows"] == 2
    assert row["dose_units"] == pytest.approx(0.5)
    assert row["dose_units_n"] == 2
    assert row["unsafe_given_recommendation_pct"] == pytest.approx(50.0)
    assert row["unsafe_given_recommendation_pct_n"] == 2


def test_oc_table_all_stopped_has_nan_conditional_metrics():
    stopped = [row for row in _stopped_results() if not row["recommendation_made"]]
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        row = _records(sdb.oc_table(stopped))[0]

    assert row["recommendation_pct"] == 0.0
    assert row["no_feasible_dose_pct"] == 100.0
    assert row["rec_unsafe_pct"] == 0.0
    for metric in ("rec_efficacy", "dose_units", "unsafe_given_recommendation_pct"):
        assert np.isnan(row[metric])
        assert np.isnan(row[f"{metric}_mcse"])
        assert row[f"{metric}_n"] == 0
        assert row[f"{metric}_n_rows"] == 0


def test_selection_table_conditions_on_recommendation_and_retains_trial_denominator():
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        rows = _records(sdb.selection_table(
            _stopped_results(), "oc_handcheck", z=0, gamma=0.7
        ))
    pct = "cEI_selected_pct"
    assert sum(row[pct] for row in rows) == pytest.approx(100.0)
    assert {row["cEI_n"] for row in rows} == {2}
    assert {row["cEI_n_rows"] for row in rows} == {2}
    assert {row["cEI_trial_n_rows"] for row in rows} == {4}
    assert {row["cEI_recommendation_pct"] for row in rows} == {50.0}


def test_selection_table_all_stopped_reports_no_fictitious_dose():
    stopped = [row for row in _stopped_results() if not row["recommendation_made"]]
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        rows = _records(sdb.selection_table(
            stopped, "oc_handcheck", z=0, gamma=0.7
        ))
    assert all(np.isnan(row["cEI_selected_pct"]) for row in rows)
    assert {row["cEI_n"] for row in rows} == {0}
    assert {row["cEI_n_rows"] for row in rows} == {0}
    assert {row["cEI_trial_n_rows"] for row in rows} == {2}
    assert {row["cEI_recommendation_pct"] for row in rows} == {0.0}


def test_oc_latex_prints_only_rounded_display(capsys):
    table = sdb.oc_table(_handchecked_results())
    latex = sdb.oc_latex(table)
    assert capsys.readouterr().out.strip() == latex
    assert "random among eligible combinations" in latex
    assert "Final selection made" in latex
    assert "Stopped, no final selection" in latex
    assert "Final selection above toxicity limit" in latex
    assert "True mean efficacy at final selection" in latex
    assert "Above-limit records or participants" in latex
    assert "Empty qualifying-set occasions" in latex
    assert "Feasibility stop" not in latex
    assert "Patients above" not in latex
    assert "Empty events" not in latex
    assert r"50.0\pm28.9" in latex
    assert r"\begin{tabular}" in latex


def test_missing_random_warns_but_does_not_fabricate_row():
    rows = [row for row in _handchecked_results() if row["policy"] == "cEI"]
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        table = sdb.oc_table(rows)
    assert [row["policy"] for row in _records(table)] == ["cEI"]
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        selection = sdb.selection_table(rows, "oc_handcheck", z=0, gamma=0.7)
    assert not any("random among eligible combinations" in name for name in _records(selection)[0])


def _unsafe_cluster_failure_case():
    """Compact reconstruction of rerun_osa unsafe counts by seed.

    The histograms are hand-checked from the current 2,000 rows per policy.  Each
    seed contributes ten rows (five gates by two strata), so this fixture preserves
    exactly the seed-level values relevant to the documented clustered MCSEs.
    """
    histograms = {
        "cEI": {0: 114, 1: 34, 2: 15, 3: 19, 4: 10, 5: 3, 6: 2, 7: 2, 8: 1},
        "cKG": {0: 44, 1: 41, 2: 37, 3: 30, 4: 21, 5: 12,
                 6: 7, 7: 3, 8: 2, 9: 2, 10: 1},
    }
    base = _handchecked_results()[0]
    rows = []
    for policy, histogram in histograms.items():
        seed = 0
        for unsafe_count, seed_count in histogram.items():
            for _ in range(seed_count):
                for cell in range(10):
                    unsafe = cell < unsafe_count
                    dose = (1.0, 1.0) if unsafe else (0.5, 0.5)
                    rows.append(dict(
                        base,
                        policy=policy,
                        seed=seed,
                        stratum=cell % 2,
                        gamma=0.5 + 0.1 * (cell // 2),
                        rec_d1=dose[0],
                        rec_d2=dose[1],
                        rec_true_eff=sum(dose),
                        rec_true_tox=sum(dose),
                        rec_unsafe=int(unsafe),
                        dose_units=float(np.linalg.norm(np.asarray(dose) - 0.5) / 0.5),
                    ))
                seed += 1
        assert seed == 200
    return rows


def test_oc_table_reproduces_rerun_osa_clustered_unsafe_mcse():
    with pytest.warns(UserWarning, match="no 'random' policy rows"):
        rows = _records(sdb.oc_table(_unsafe_cluster_failure_case()))
    cei = next(row for row in rows if row["policy"] == "cEI")
    ckg = next(row for row in rows if row["policy"] == "cKG")
    assert (cei["n"], cei["n_rows"]) == (200, 2000)
    assert (ckg["n"], ckg["n_rows"]) == (200, 2000)
    assert cei["rec_unsafe_pct_mcse"] == pytest.approx(1.136, abs=0.0005)
    assert ckg["rec_unsafe_pct_mcse"] == pytest.approx(1.457, abs=0.0005)


def test_oc_reports_refuse_mixed_mode_and_kap():
    rows = _handchecked_results()
    rows[0] = dict(rows[0], mode="predictive")
    with pytest.raises(ValueError, match="different 'mode'"):
        sdb.oc_table(rows)
    with pytest.raises(ValueError, match="different 'mode'"):
        sdb.selection_table(rows, "oc_handcheck", z=0, gamma=0.7)

    rows = _handchecked_results()
    rows[0] = dict(rows[0], kap=2.0)
    with pytest.raises(ValueError, match="different 'kap'"):
        sdb.oc_table(rows)


def test_selection_filters_design_and_clusters_by_distinct_seed():
    rows = _handchecked_results()
    rows += [dict(row, gamma=0.8) for row in rows]
    table = sdb.selection_table(
        rows, "oc_handcheck", z=0, gamma=0.7, mode="latent", noise="fixed",
        kap=1.0, budget=10, r_k=2, grid_n=3, warmup=2, empty_gate="pf",
    )
    first = _records(table)[0]
    assert first["cEI_n"] == 4
    assert first["cEI_n_rows"] == 4


def test_shipped_demo_records_feed_both_oc_reports():
    path = Path(__file__).parent / "fixtures" / "historical_demo_results.json"
    rows = json.loads(path.read_text())
    oc = _records(sdb.oc_table(rows, sim="osa", gamma=0.7, kap=1.0))
    assert {(row["policy"], row["n"], row["n_rows"]) for row in oc} == {
        ("cEI", 5, 5), ("random", 5, 5),
    }
    selection = _records(sdb.selection_table(rows, "osa", z=0, gamma=0.7, kap=1.0))
    for policy in ("cEI", "random among eligible combinations"):
        assert sum(row[f"{policy}_selected_pct"] for row in selection) == pytest.approx(100.0)


def test_oc_reports_reject_inconsistent_saved_fields():
    rows = _handchecked_results()
    rows[0] = dict(rows[0], n_gate_pass=1, n_gate_pass_safe=2)
    with pytest.raises(ValueError, match="gate counts"):
        sdb.oc_table(rows)

    rows = _handchecked_results()
    rows[0] = dict(rows[0], rec_d1=0.4)
    with pytest.raises(ValueError, match="not on the reported"):
        sdb.selection_table(rows, "oc_handcheck", z=0, gamma=0.7)


def test_oc_reports_fall_back_to_records_without_pandas(monkeypatch):
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas deliberately unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    assert isinstance(sdb.oc_table(_handchecked_results()), list)
    assert isinstance(sdb.selection_table(
        _handchecked_results(), "oc_handcheck", z=0, gamma=0.7
    ), list)

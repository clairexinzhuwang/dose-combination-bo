import dose_combination_bo as sdb


def _records():
    design = dict(
        sim="osa", stratum=0, gamma=0.7, mode="latent", noise="fixed",
        kap=1.0, budget=40, r_k=2, grid_n=5, warmup=4, empty_gate="pf",
        protocol_scaffold="start_low_expansion", region_step=0.25,
        empty_gate_stop_after=3, exclude_repeats_during_expansion=True,
    )
    return [
        dict(design, policy="cKG", seed=0, recommendation_made=True,
             dose_units=1.0, rec_unsafe=1),
        dict(design, policy="cKG", seed=1, recommendation_made=False,
             dose_units=None, rec_unsafe=None),
        dict(design, policy="cEI-tMSE", seed=0, recommendation_made=True,
             dose_units=2.0, rec_unsafe=0),
        dict(design, policy="cEI-tMSE", seed=1, recommendation_made=False,
             dose_units=None, rec_unsafe=None),
    ]


def test_leaderboard_excludes_stops_for_conditional_metric():
    ranked = sdb.leaderboard(
        _records(), metric="dose_units", recommendation_conditional=True
    )
    assert [(row[0], row[1], row[3]) for row in ranked] == [
        ("cKG", 1.0, 1),
        ("cEI-tMSE", 2.0, 1),
    ]


def test_leaderboard_counts_stop_as_no_unsafe_recommendation():
    ranked = sdb.leaderboard(_records(), metric="rec_unsafe")
    values = {policy: mean for policy, mean, _, _ in ranked}
    assert values == {"cEI-tMSE": 0.0, "cKG": 0.5}

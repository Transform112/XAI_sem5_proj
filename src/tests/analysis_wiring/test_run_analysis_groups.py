"""T49 follow-up tests: multi-group CLI resolution for the S8 real wiring."""

from __future__ import annotations

import pytest


def test_seed42_all_resolves_six_groups_in_reference_group_order():
    from scripts.run_analysis import resolve_groups

    groups = resolve_groups("seed42-all", None, None, 42)
    assert groups == [
        ("ft", "egraphsage", 42),
        ("ft", "gcn", 42),
        ("replay", "egraphsage", 42),
        ("replay", "gcn", 42),
        ("ewc", "egraphsage", 42),
        ("ewc", "gcn", 42),
    ]
    # The first group is the noise reference group (ft/egraphsage/seed42).
    assert groups[0] == ("ft", "egraphsage", 42)


def test_cartesian_strategies_models_matches_seed42_all():
    from scripts.run_analysis import resolve_groups

    assert resolve_groups(None, "ft,replay,ewc", "egraphsage,gcn", 42) == resolve_groups(
        "seed42-all", None, None, 42
    )


def test_explicit_group_list_and_per_group_seed():
    from scripts.run_analysis import resolve_groups

    groups = resolve_groups("ft:egraphsage:42,replay:gcn:43,ewc:gcn:42", None, None, 42)
    assert groups == [
        ("ft", "egraphsage", 42),
        ("replay", "gcn", 43),
        ("ewc", "gcn", 42),
    ]
    # duplicates are dropped (stable order)
    assert resolve_groups("ft:gcn:42,ft:gcn:42", None, None, 42) == [("ft", "gcn", 42)]


def test_legacy_single_group_returns_none():
    from scripts.run_analysis import resolve_groups

    assert resolve_groups(None, None, None, 42) is None


@pytest.mark.parametrize("groups_arg", ["", "ft:bad:42", "nope:egraphsage:42", "ft:gcn:oops"])
def test_invalid_group_entries_raise(groups_arg):
    from scripts.run_analysis import resolve_groups

    with pytest.raises(ValueError):
        resolve_groups(groups_arg, None, None, 42)

import asyncio

import pytest

from app.benchmark.tree_search import WeightedChild, forecast_tree


def terminal(state):
    return state if state in {"yes", "no", "unknown"} else None


def evaluate(tree, *, depth=2, search_budget=0, evaluation_budget=0, fallback_probability=.25, seed=0):
    calls = []

    async def expand(state):
        calls.append(state)
        return [WeightedChild(child, probability) for child, probability in tree[state]]

    result = asyncio.run(forecast_tree(
        "root", expand=expand, terminal=terminal, depth=depth,
        search_budget=search_budget, evaluation_budget=evaluation_budget,
        fallback_probability=fallback_probability, seed=seed,
    ))
    assert len(calls) == result.search_expansions + result.evaluation_expansions
    assert result.search_expansions <= search_budget
    assert result.evaluation_expansions <= evaluation_budget
    assert sum(row.mass for row in result.frontier) == pytest.approx(1)
    for first in result.frontier:
        for second in result.frontier:
            if first != second:
                assert first.path != second.path[:len(first.path)]
    return result, calls


def test_full_tree_uses_original_transition_mass_and_distinguishes_unknown():
    result, _ = evaluate(
        {"root": [("yes", .2), ("sub", .8)], "sub": [("yes", .5), ("no", .3), ("unknown", .2)]},
        search_budget=8,
    )
    assert result.p_yes == pytest.approx(.2 + .8 * .5 + .8 * .2 * .25)
    assert result.covered_mass == 1 and result.uncovered_mass == 0
    assert result.unknown_mass == pytest.approx(.16)
    assert result.resolved_mass == pytest.approx(.84)
    assert result.evaluation_rollouts == 0
    assert result.frontier_size == 4


def test_unexpanded_frontier_reuses_only_the_prespecified_fallback():
    result, calls = evaluate({"root": [("yes", .2), ("sub", .8)]}, search_budget=1)
    assert calls == ["root"]
    assert result.p_yes == pytest.approx(.2 + .8 * .25)
    assert result.covered_mass == pytest.approx(.2)
    assert result.uncovered_mass == pytest.approx(.8)
    assert result.unknown_mass == 0


def test_fresh_conditional_rollouts_reproduce_and_target_the_policy_distribution():
    tree = {"root": [("yes", .2), ("sub", .8)], "sub": [("yes", .5), ("no", .5)]}
    options = dict(search_budget=2, evaluation_budget=1000, seed=19)
    result, calls = evaluate(tree, **options)
    repeated, _ = evaluate(tree, **options)
    assert repeated == result
    assert calls[:result.search_expansions] == ["root", "sub"]
    assert calls[result.search_expansions:] == ["sub"] * 1000
    assert result.evaluation_rollouts == 1000
    assert result.p_yes == pytest.approx(.6, abs=.04)


def test_fixed_evaluation_allocation_cannot_spend_early_stop_savings():
    tree = {"root": [("yes", 1.0)]}
    result, calls = evaluate(tree, depth=2, evaluation_budget=5)
    assert result.p_yes == 1
    assert result.evaluation_rollouts == 2
    assert calls == ["root", "root"]


def test_depth_limit_records_unknown_without_making_extra_expansions():
    result, calls = evaluate({"root": [("sub", 1.0)]}, depth=1, search_budget=1, evaluation_budget=10)
    assert calls == ["root"]
    assert result.p_yes == .25 and result.unknown_mass == 1
    assert result.uncovered_mass == 0


@pytest.mark.parametrize("weights", [[.4, .4], [-.1, 1.1], [float("nan"), 1]])
def test_invalid_transition_probability_is_rejected(weights):
    tree = {"root": [("yes", weights[0]), ("no", weights[1])]}
    with pytest.raises(ValueError, match="probabilities"):
        evaluate(tree, search_budget=1)

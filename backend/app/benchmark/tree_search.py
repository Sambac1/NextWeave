"""Bounded UCT search and fresh probability estimates on its frozen frontier."""

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar


State = TypeVar("State")
Outcome = Literal["yes", "no", "unknown"]


@dataclass(frozen=True)
class WeightedChild(Generic[State]):
    state: State
    probability: float


@dataclass(frozen=True)
class FrontierEstimate:
    path: tuple[int, ...]
    mass: float
    terminal: Outcome | None
    rollouts: int
    conditional_p_yes: float | None
    conditional_unknown: float | None


@dataclass(frozen=True)
class TreeForecast:
    p_yes: float
    covered_mass: float
    uncovered_mass: float
    unknown_mass: float
    resolved_mass: float
    search_expansions: int
    evaluation_expansions: int
    frontier_size: int
    evaluation_rollouts: int
    frontier: tuple[FrontierEstimate, ...]


@dataclass
class _Node(Generic[State]):
    state: State
    path: tuple[int, ...]
    mass: float
    outcome: Outcome | None
    children: list["_Node[State]"] = field(default_factory=list)
    visits: int = 0
    reward: float = 0.0
    open: bool = True


class _ExpansionBudget(Generic[State]):
    def __init__(self, limit: int, expand: Callable[[State], Awaitable[Sequence[WeightedChild[State]]]]):
        self.limit = limit
        self.expand = expand
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    async def children(self, state: State) -> list[WeightedChild[State]]:
        if not self.remaining:
            raise RuntimeError("Tree expansion budget exhausted")
        self.used += 1
        children = list(await self.expand(state))
        if any(not math.isfinite(child.probability) or child.probability < 0 for child in children):
            raise ValueError("Tree transitions require nonnegative, finite probabilities")
        total = math.fsum(child.probability for child in children)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Tree transition probabilities must sum to one")
        return [WeightedChild(child.state, child.probability / total) for child in children if child.probability > 0]


def _outcome(state, depth_left, terminal):
    outcome = terminal(state)
    if outcome not in (None, "yes", "no", "unknown"):
        raise ValueError("Tree terminal evaluator returned an invalid outcome")
    return outcome if depth_left or outcome is not None else "unknown"


def _uct(node):
    candidates = [child for child in node.children if child.open]
    unseen = [child for child in candidates if child.visits == 0]
    if unseen:
        return max(unseen, key=lambda child: child.mass)
    return max(candidates, key=lambda child: (
        child.reward / child.visits + math.sqrt(2 * math.log(max(2, node.visits)) / child.visits),
        child.mass,
    ))


def _reward(outcome, fallback):
    if outcome == "yes":
        return 1.0
    if outcome == "no":
        return 0.0
    return fallback


async def _rollout(state, depth_left, terminal, budget, rng):
    outcome = _outcome(state, depth_left, terminal)
    while outcome is None and budget.remaining:
        children = await budget.children(state)
        chosen = rng.choices(children, weights=[child.probability for child in children])[0]
        state = chosen.state
        depth_left -= 1
        outcome = _outcome(state, depth_left, terminal)
    return outcome


def _frontier(root):
    stack = [root]
    leaves = []
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(reversed(node.children))
        else:
            leaves.append(node)
    return leaves


def _allocate_rollouts(frontier, depth, budget):
    # Reserve worst-case expansion cost before observing any evaluation outcome.
    # Observed early stops cannot buy extra draws and change the estimator.
    pending = sorted((node for node in frontier if node.outcome is None), key=lambda node: -node.mass)
    costs = {node.path: depth - len(node.path) for node in pending}
    counts = {node.path: 0 for node in pending}
    while budget and pending:
        pending = [node for node in pending if costs[node.path] <= budget]
        if pending:
            best = max(pending, key=lambda node: (counts[node.path] == 0, node.mass / (counts[node.path] + 1)))
            counts[best.path] += 1
            budget -= costs[best.path]
    return counts


async def forecast_tree(
    root: State,
    *,
    expand: Callable[[State], Awaitable[Sequence[WeightedChild[State]]]],
    terminal: Callable[[State], Outcome | None],
    depth: int,
    search_budget: int,
    evaluation_budget: int,
    fallback_probability: float,
    seed: int = 0,
) -> TreeForecast:
    """Estimate the Yes probability under the original transition distribution.

    Both budgets count calls to the expander. States should be immutable. Use a
    reproducible expander for a fixed model tree, including all chance branches.
    Search data selects strata, but probability estimates use fresh rollouts.
    A terminal callback returns None for unresolved states that can still evolve.
    """
    for name, value in (("depth", depth), ("search_budget", search_budget), ("evaluation_budget", evaluation_budget)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 <= fallback_probability <= 1:
        raise ValueError("Fallback probability must be in [0, 1]")

    root_outcome = _outcome(root, depth, terminal)
    tree = _Node(root, (), 1.0, root_outcome, open=root_outcome is None)
    search = _ExpansionBudget(search_budget, expand)
    search_rng = random.Random(f"{seed}:search")
    while tree.open and search.remaining:
        node = tree
        path = [node]
        while node.children:
            node = _uct(node)
            path.append(node)
        children = await search.children(node.state)
        depth_left = depth - len(node.path) - 1
        for index, child in enumerate(children):
            outcome = _outcome(child.state, depth_left, terminal)
            node.children.append(_Node(child.state, (*node.path, index), node.mass * child.probability,
                                       outcome, open=outcome is None))
        chosen = search_rng.choices(node.children, weights=[child.probability for child in children])[0]
        outcome = chosen.outcome
        if outcome is None:
            outcome = await _rollout(chosen.state, depth_left, terminal, search, search_rng)
        reward = _reward(outcome, fallback_probability)
        chosen.visits += 1
        chosen.reward += reward
        for ancestor in reversed(path):
            ancestor.visits += 1
            ancestor.reward += reward
            ancestor.open = any(child.open for child in ancestor.children)

    frontier = _frontier(tree)
    counts = _allocate_rollouts(frontier, depth, evaluation_budget)
    evaluate = _ExpansionBudget(evaluation_budget, expand)
    evaluation_rng = random.Random(f"{seed}:evaluate")
    estimates = []
    yes_mass = unknown_mass = uncovered_mass = 0.0
    for node in frontier:
        count = counts.get(node.path, 0)
        if node.outcome is not None:
            p_yes = float(node.outcome == "yes")
            p_unknown = float(node.outcome == "unknown")
        elif count:
            outcomes = [await _rollout(node.state, depth - len(node.path), terminal, evaluate, evaluation_rng)
                        for _ in range(count)]
            p_yes = outcomes.count("yes") / count
            p_unknown = outcomes.count("unknown") / count
        else:
            p_yes = p_unknown = None
        if p_yes is None:
            uncovered_mass += node.mass
        else:
            yes_mass += node.mass * p_yes
            unknown_mass += node.mass * p_unknown
        estimates.append(FrontierEstimate(node.path, node.mass, node.outcome, count, p_yes, p_unknown))

    covered_mass = math.fsum(estimate.mass for estimate in estimates if estimate.conditional_p_yes is not None)
    return TreeForecast(
        p_yes=min(1.0, yes_mass + (unknown_mass + uncovered_mass) * fallback_probability),
        covered_mass=min(1.0, covered_mass),
        uncovered_mass=min(1.0, uncovered_mass),
        unknown_mass=min(1.0, unknown_mass),
        resolved_mass=max(0.0, min(1.0, covered_mass - unknown_mass)),
        search_expansions=search.used,
        evaluation_expansions=evaluate.used,
        frontier_size=len(estimates),
        evaluation_rollouts=sum(counts.values()),
        frontier=tuple(estimates),
    )


@dataclass
class SearchNode(Generic[State]):
    path: tuple[int, ...]
    state: State | None
    mass: float
    conditional_p_yes: float | None = None
    visits: int = 0
    value: float = 0.0
    children: list["SearchNode[State]"] = field(default_factory=list)


def _can_expand(node, depth):
    if node.state is None or len(node.path) >= depth:
        return False
    return not node.children or any(_can_expand(child, depth) for child in node.children)


async def search_probability_tree(state, expand, *, depth, iterations, fallback):
    """PUCT on plausible, uncertain states. Only frontier probability mass enters the estimate."""
    root = SearchNode((), state, 1.0)
    stopped = None
    for _ in range(iterations):
        if not _can_expand(root, depth):
            break
        path = [root]
        while path[-1].children:
            parent = path[-1]
            path.append(max(
                (child for child in parent.children if _can_expand(child, depth)),
                key=lambda child: (child.value / max(child.visits, 1)
                                   + (child.mass / parent.mass) * math.sqrt(parent.visits + 1) / (1 + child.visits),
                                   child.mass),
            ))
        leaf = path[-1]
        try:
            transitions = await expand(leaf.path, leaf.state)
        except Exception as exc:
            stopped = f"{type(exc).__name__}: {exc}"
            break
        total = math.fsum(row[0] for row in transitions)
        if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-9) or any(
            not math.isfinite(weight) or weight <= 0 or (probability is not None and not 0 <= probability <= 1)
            for weight, probability, _ in transitions
        ):
            raise ValueError("Invalid MCTS branch partition")
        for index, (weight, probability, child_state) in enumerate(transitions, 1):
            leaf.children.append(SearchNode((*leaf.path, index), child_state, leaf.mass * weight / total, probability))
        uncertainty = sum((child.mass / leaf.mass) * (4 * child.conditional_p_yes * (1 - child.conditional_p_yes))
                          for child in leaf.children if child.conditional_p_yes is not None)
        for node in path:
            node.visits += 1
            node.value += uncertainty

    nodes = []
    for node in _walk(root):
        nodes.append({"path": list(node.path), "depth": len(node.path), "mass": node.mass, "visits": node.visits,
                      "search_value": node.value / node.visits if node.visits else None,
                      "conditional_p_yes": node.conditional_p_yes, "frontier": not node.children,
                      "contribution": node.mass * node.conditional_p_yes
                      if not node.children and node.conditional_p_yes is not None else None})
    residual = math.fsum(node["mass"] for node in nodes if node["frontier"] and node["contribution"] is None)
    known_yes = math.fsum(node["contribution"] for node in nodes if node["contribution"] is not None)
    return {
        "p_yes": min(1.0, known_yes + residual * fallback) if residual < 1 - 1e-12 else None,
        "coverage": max(0.0, 1 - residual), "unexplored_mass": residual,
        "unknown_bounds": [known_yes, min(1.0, known_yes + residual)],
        "search_iterations": root.visits, "tree": nodes, "error": stopped,
    }


def _walk(root):
    yield root
    for child in root.children:
        yield from _walk(child)

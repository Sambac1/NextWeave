"""Offline evaluation of forecasts against a fixed cohort of resolved contracts."""

import csv
import io
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, model_validator


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    as_of: AwareDatetime
    observed_yes: StrictBool
    fallback_probability: float = Field(ge=0, le=1, strict=True)
    market_probability: float | None = Field(default=None, ge=0, le=1, strict=True)
    market_observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def verify_market_snapshot(self):
        if (self.market_probability is None) != (self.market_observed_at is None):
            raise ValueError("A market probability requires its observation timestamp")
        if self.market_observed_at is not None and self.market_observed_at > self.as_of:
            raise ValueError("Market price was observed after the forecast cutoff")
        return self

    @property
    def key(self):
        return self.case_id, self.as_of


class EvaluationForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    as_of: AwareDatetime
    method: str = Field(min_length=1)
    p_yes: float | None = Field(default=None, ge=0, le=1, strict=True)
    fallback_used: StrictBool = False
    calls: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @property
    def key(self):
        return self.case_id, self.as_of


def _brier(probability, case):
    return (probability - case.observed_yes) ** 2


def _log_loss(probability, case):
    epsilon = 1e-15
    clipped = max(epsilon, min(1 - epsilon, probability))
    return -math.log(clipped) if case.observed_yes else -math.log1p(-clipped)


def _calibration_bins(cases, probabilities):
    bins = defaultdict(list)
    for case, probability in zip(cases, probabilities):
        bins[min(4, int(probability * 5))].append((probability, case.observed_yes))
    return [
        {
            "lower": bucket / 5,
            "upper": (bucket + 1) / 5,
            "count": len(values),
            "mean_p_yes": mean(p for p, _ in values),
            "observed_yes_rate": mean(y for _, y in values),
        }
        for bucket, values in sorted(bins.items())
    ]


def _percentile(ordered, fraction):
    index = fraction * (len(ordered) - 1)
    left = math.floor(index)
    right = math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def _paired(cases, predictions, reference, samples, seed):
    differences = defaultdict(list)
    for case, p, q in zip(cases, predictions, reference):
        differences[case.group_id].append(_brier(p, case) - _brier(q, case))
    clusters = [(sum(values), len(values)) for values in differences.values()]
    interval = None
    if samples and len(clusters) > 1:
        rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            selected = rng.choices(clusters, k=len(clusters))
            draws.append(sum(total for total, _ in selected) / sum(n for _, n in selected))
        draws.sort()
        interval = [_percentile(draws, .025), _percentile(draws, .975)]
    return {
        "case_count": len(cases),
        "event_count": len(clusters),
        "delta_brier": sum(total for total, _ in clusters) / len(cases),
        "confidence_interval_95": interval,
    }


def score_forecasts(
    cases: Sequence[EvaluationCase],
    forecasts: Sequence[EvaluationForecast],
    *,
    methods: Sequence[str],
    reference_method: str,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """Score all prespecified cases, replacing unavailable probabilities with the frozen prior.

    A pair is identified by market and cutoff. Event clusters retain all of their
    contracts and cutoffs during bootstrap resampling.
    """
    if not cases:
        raise ValueError("At least one resolved case is required")
    if len(methods) != len(set(methods)) or reference_method not in methods:
        raise ValueError("Methods must be unique and include the reference method")
    if not 0 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be between 0 and 10000")
    # Stable ordering makes the bootstrap reproducible when files arrive in a different order.
    cases = sorted(cases, key=lambda case: (case.group_id, case.key))
    cohort = {case.key: case for case in cases}
    if len(cohort) != len(cases):
        raise ValueError("Duplicate case and cutoff")
    contracts = {}
    for case in cases:
        target = case.group_id, case.observed_yes
        if contracts.setdefault(case.case_id, target) != target:
            raise ValueError("A contract must keep the same event group and outcome across cutoffs")
    records = {}
    for row in forecasts:
        if row.key not in cohort or row.method not in methods:
            raise ValueError("Forecast does not match the fixed cohort, cutoff, or methods")
        key = row.method, row.key
        if key in records:
            raise ValueError("Duplicate forecast for the same method, case, and cutoff")
        records[key] = row

    probabilities = {}
    metrics = {}
    available = {}
    for method in methods:
        rows = [records.get((method, case.key)) for case in cases]
        covered = [row is not None and row.p_yes is not None and not row.fallback_used for row in rows]
        available[method] = covered
        values = [row.p_yes if usable else case.fallback_probability
                  for case, row, usable in zip(cases, rows, covered)]
        probabilities[method] = values
        observed_losses = [_brier(p, case) for case, p, usable in zip(cases, values, covered) if usable]
        bins = _calibration_bins(cases, values)
        metrics[method] = {
            "brier": mean(_brier(p, case) for case, p in zip(cases, values)),
            "log_loss": mean(_log_loss(p, case) for case, p in zip(cases, values)),
            "coverage": sum(covered) / len(cases),
            "fallback_count": len(cases) - sum(covered),
            "covered_only_brier": mean(observed_losses) if observed_losses else None,
            "calibration_bins": bins,
            "ece": sum(b["count"] * abs(b["mean_p_yes"] - b["observed_yes_rate"]) for b in bins) / len(cases),
            "failed_forecasts": len(cases) - sum(covered),
        }
        for field in ("calls", "tokens", "elapsed_seconds"):
            reported = [getattr(row, field) for row in rows if row is not None and getattr(row, field) is not None]
            metrics[method][f"mean_{field}"] = mean(reported) if reported else None
            metrics[method][f"{field}_count"] = len(reported)

    common = [index for index in range(len(cases)) if all(available[method][index] for method in methods)]
    common_metrics = {
        method: {"brier": mean(_brier(values[index], cases[index]) for index in common),
                 "log_loss": mean(_log_loss(values[index], cases[index]) for index in common)}
        for method, values in probabilities.items()
    } if common else {}

    paired = {
        method: _paired(cases, values, probabilities[reference_method], bootstrap_samples, seed)
        for method, values in probabilities.items()
        if method != reference_method
    }
    price_indices = [index for index, case in enumerate(cases) if case.market_probability is not None]
    market_comparison = None
    if price_indices:
        priced = [cases[index] for index in price_indices]
        market = [case.market_probability for case in priced]
        market_comparison = {
            "case_count": len(priced),
            "brier": mean(_brier(p, case) for case, p in zip(priced, market)),
            "log_loss": mean(_log_loss(p, case) for case, p in zip(priced, market)),
            "paired": {
                method: _paired(priced, [values[index] for index in price_indices], market, bootstrap_samples, seed)
                for method, values in probabilities.items()
            },
        }
    return {
        "case_count": len(cases),
        "event_count": len({case.group_id for case in cases}),
        "reference_method": reference_method,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "score_version": "binary-brier-logloss-ece5-v1",
        "log_loss_epsilon": 1e-15,
        "metrics": metrics,
        "common_completed": {"case_count": len(common), "metrics": common_metrics},
        "paired": paired,
        "market_comparison": market_comparison,
    }


def metrics_csv(report: dict) -> str:
    """Export scalar metrics, costs and paired estimates for the paper tables."""
    fields = (
        "brier", "log_loss", "ece", "coverage", "fallback_count", "failed_forecasts", "covered_only_brier",
        "mean_calls", "calls_count", "mean_tokens", "tokens_count", "mean_elapsed_seconds", "elapsed_seconds_count",
    )
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "method", "case_count", "event_count", *fields, "reference_method",
        "delta_brier", "delta_brier_ci_low", "delta_brier_ci_high",
        "market_case_count", "market_brier", "delta_market_brier", "delta_market_ci_low", "delta_market_ci_high",
        "common_case_count", "common_brier", "common_log_loss",
    ])
    market = report["market_comparison"] or {}
    common = report.get("common_completed", {})
    for method, metrics in report["metrics"].items():
        paired = report["paired"].get(method, {})
        market_pair = market.get("paired", {}).get(method, {})
        common_metric = common.get("metrics", {}).get(method, {})
        writer.writerow([
            method, report["case_count"], report["event_count"], *(metrics[field] for field in fields),
            report["reference_method"], paired.get("delta_brier"),
            *(paired.get("confidence_interval_95") or (None, None)),
            market.get("case_count"), market.get("brier"), market_pair.get("delta_brier"),
            *(market_pair.get("confidence_interval_95") or (None, None)),
            common.get("case_count"), common_metric.get("brier"), common_metric.get("log_loss"),
        ])
    return output.getvalue()

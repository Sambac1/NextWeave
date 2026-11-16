import csv
import io
import math

import pytest
from pydantic import ValidationError

from app.benchmark.scoring import EvaluationCase, EvaluationForecast, metrics_csv, score_forecasts


CUTOFF = "2026-04-07T12:00:00Z"


def case(identifier, group, observed, *, price=None, as_of=CUTOFF):
    return EvaluationCase(
        case_id=identifier,
        group_id=group,
        as_of=as_of,
        observed_yes=observed,
        fallback_probability=.25,
        market_probability=price,
        market_observed_at="2026-04-07T11:30:00Z" if price is not None else None,
    )


def forecast(identifier, method, probability, **metadata):
    return EvaluationForecast(case_id=identifier, as_of=CUTOFF, method=method, p_yes=probability, **metadata)


def test_paired_report_scores_fixed_cohort_and_exports_results():
    cases = [case("m1", "election", True, price=.6), case("m2", "election", False), case("m3", "deal", True)]
    predictions = [
        forecast("m1", "qa", .5, calls=1, tokens=40),
        forecast("m2", "qa", .2, calls=1, tokens=50),
        forecast("m3", "qa", .8, calls=1, tokens=60),
        forecast("m1", "sample", .9, calls=7, tokens=100),
        forecast("m2", "sample", .8, calls=3, tokens=20, fallback_used=True),
        # Missing m3 must still count in scoring and coverage.
    ]
    options = dict(methods=["qa", "sample"], reference_method="qa", bootstrap_samples=250, seed=7)
    report = score_forecasts(cases, predictions, **options)
    assert report == score_forecasts(list(reversed(cases)), list(reversed(predictions)), **options)
    expected_brier = (.1 ** 2 + .25 ** 2 + .75 ** 2) / 3
    assert report["event_count"] == 2
    assert report["metrics"]["sample"]["brier"] == pytest.approx(expected_brier)
    assert report["metrics"]["sample"]["log_loss"] == pytest.approx(-math.log(.9 * .75 * .25) / 3)
    assert report["metrics"]["sample"]["coverage"] == pytest.approx(1 / 3)
    common = report["common_completed"]
    assert common["case_count"] == 1
    assert common["metrics"]["sample"]["brier"] == pytest.approx(.01)
    assert common["metrics"]["qa"]["brier"] == pytest.approx(.25)
    assert report["metrics"]["sample"]["fallback_count"] == 2
    assert report["metrics"]["sample"]["calls_count"] == 2
    assert report["metrics"]["sample"]["mean_tokens"] == 60
    assert report["metrics"]["sample"]["calibration_bins"][0]["mean_p_yes"] == .25
    assert report["metrics"]["sample"]["calibration_bins"][0]["observed_yes_rate"] == .5
    pair = report["paired"]["sample"]
    assert pair["delta_brier"] == pytest.approx(expected_brier - (.5 ** 2 + .2 ** 2 + .2 ** 2) / 3)
    assert len(pair["confidence_interval_95"]) == 2
    market = report["market_comparison"]
    assert market["case_count"] == 1
    assert market["brier"] == pytest.approx(.16)
    assert market["paired"]["sample"]["delta_brier"] == pytest.approx(-.15)
    assert market["paired"]["sample"]["confidence_interval_95"] is None
    rows = list(csv.DictReader(io.StringIO(metrics_csv(report))))
    assert [row["method"] for row in rows] == ["qa", "sample"]
    assert rows[1]["market_case_count"] == "1"
    assert float(rows[1]["delta_market_brier"]) == pytest.approx(-.15)
    assert float(rows[1]["common_brier"]) == pytest.approx(.01)


def test_rejects_mismatched_cutoffs_and_duplicate_forecasts():
    observations = [case("m1", "election", False)]
    predictions = [forecast("m1", "qa", 0)]
    config = dict(methods=["qa"], reference_method="qa")
    with pytest.raises(ValueError, match="Duplicate forecast"):
        score_forecasts(observations, predictions * 2, **config)
    with pytest.raises(ValueError, match="Duplicate case"):
        score_forecasts(observations * 2, predictions, **config)
    with pytest.raises(ValueError, match="cutoff"):
        score_forecasts([case("m1", "election", False, as_of="2026-04-01T00:00:00Z")], predictions, **config)


def test_market_comparison_requires_point_in_time_prices():
    with pytest.raises(ValidationError, match="after the forecast cutoff"):
        EvaluationCase(
            case_id="m1", group_id="g", observed_yes=True, as_of=CUTOFF,
            fallback_probability=.25, market_probability=.5, market_observed_at="2026-04-08T12:00:00Z",
        )
    with pytest.raises(ValidationError, match="observation timestamp"):
        EvaluationCase(
            case_id="m1", group_id="g", observed_yes=True, as_of=CUTOFF, fallback_probability=.25,
            market_probability=.5,
        )


def test_cross_cutoff_rows_remain_one_cluster_and_share_the_terminal_label():
    first = case("m1", "election", False)
    second = case("m1", "election", False, as_of="2026-04-01T00:00:00Z")
    predictions = [forecast("m1", "qa", 0), forecast("m1", "sample", 1)]
    options = dict(methods=["qa", "sample"], reference_method="qa")
    report = score_forecasts([first, second], predictions, **options)
    pair = report["paired"]["sample"]
    assert pair["case_count"] == 2 and pair["event_count"] == 1
    assert pair["delta_brier"] == .5 and pair["confidence_interval_95"] is None
    assert math.isfinite(report["metrics"]["sample"]["log_loss"])
    with pytest.raises(ValueError, match="same event group and outcome"):
        score_forecasts([first, second.model_copy(update={"observed_yes": True})], predictions, **options)

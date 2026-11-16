from app.knowledge.evidence import validate_event, locate_quote


def check(text, quote, timestamp, **extra):
    return validate_event(
        {
            "description": "An event",
            "timestamp": timestamp,
            "evidence_text": quote,
            **extra,
        },
        {"path": "source_01/Chronology", "source_title": "Case", "content": text},
    )


def test_normalized_quote_keeps_original_offsets():
    source = "2024年3月，  工厂\n恢复了生产。"
    span = locate_quote(source, "工厂 恢复了生产。")
    assert source[slice(*span)] == "工厂\n恢复了生产。"


def test_quarter_and_effective_dates_are_not_historical_occurrences():
    quote = "The bank estimated reduced output for the first quarter of 2022."
    value, reason = check(
        "On November 17, 2021, workers voted. " + quote, quote, "2022-01"
    )
    assert value is None and "coverage" in reason
    value, _ = check(
        "On April 1, 2024, the company will begin layoffs.",
        "On April 1, 2024, the company will begin layoffs.",
        "2024-04-01",
        modality="scheduled",
    )
    assert value is None
    value, _ = check(
        "2024-04-01: a change.",
        "2024-04-01: a change.",
        "2024-04-01",
        time_role="effective_date",
    )
    assert value is None


def test_explicit_years_are_not_reanchored_and_money_is_not_a_year():
    quote = "March 2022 and April 2023 had separate strikes."
    value, _ = validate_event(
        {"timestamp": "2021-03", "description": "Strike", "evidence_text": quote},
        {"path": "source_01/2021 case", "source_title": "2021 case", "content": quote},
    )
    assert value is None
    quote = "Workers received 2024 dollars in March."
    assert check(quote, quote, "2024-03")[0] is None


def test_relative_cross_year_day_has_a_real_anchor():
    quote = "The next day, they signed the contract."
    value, _ = check("On December 31, 2023, they met. " + quote, quote, "2024-01-01")
    assert value["timestamp"] == "2024-01-01"
    assert check(quote, quote, "2024-01-01")[0] is None


def test_nearby_unrelated_sentence_cannot_supply_occurrence_date():
    quote = "The company closed the factory."
    assert (
        check(
            quote + " On March 1, 2024, the union filed a complaint.", quote, "2024-03"
        )[0]
        is None
    )
    assert (
        check(
            "On March 1, 2024, the union filed a complaint. " + quote, quote, "2024-03"
        )[0]
        is None
    )


def test_temporal_bound_and_invalid_day_are_not_accepted_as_exact_dates():
    quote = "Before March 2024, the factory closed."
    assert check(quote, quote, "2024-03")[0] is None
    quote = "On February 30, 2024, the factory closed."
    assert check(quote, quote, "2024-02-30")[0] is None


def test_announcement_is_bound_to_announcement_clause():
    quote = "On March 1, 2024, the company announced that layoffs would take effect on April 1, 2024."
    value, _ = check(
        quote, quote, "2024-04-01", time_role="announcement", modality="announced"
    )
    assert value["timestamp"][:7] == "2024-03"


def test_month_precision_can_be_inferred_from_a_document_year_but_not_a_generated_description():
    quote = "In March, workers signed the contract."
    value, _ = validate_event(
        {"timestamp": "2024-03-15", "description": "Signed", "evidence_text": quote},
        {"path": "source_01/2024", "content": quote},
    )
    assert value["timestamp"] == "2024-03"
    assert (
        check(
            "Workers signed.", "Workers signed.", "2024-03", date_original="March 2024"
        )[0]
        is None
    )


def test_dated_headings_recover_the_source_day_and_relative_occurrence():
    quote = "The navy reported that it intercepted two boats the day before."
    section = {"path": "source_01/2026/September/1 September", "source_title": "2025–2026 chronology", "content": quote}
    raw = {"description": "The navy intercepted two boats.", "timestamp": "2025-09-01",
           "evidence_text": quote, "date_original": "the day before"}
    occurrence, _ = validate_event(raw, section)
    assert occurrence["timestamp"] == "2026-08-31"
    assert occurrence["date_evidence"]["source_date_heading"] == "1 September"
    announcement, _ = validate_event(raw | {"time_role": "announcement", "modality": "announced"}, section)
    assert announcement["timestamp"] == "2026-09-01"
    assert validate_event(raw, section | {"path": "source_01/2026/September"})[0] is None

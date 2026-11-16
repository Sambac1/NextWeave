"""Assemble a report from the saved public settlement, with no model call."""
import math

from app.simulation.question_sampling import summarize_sequential


def report_view(run_id, store, config, report):
    """Expose interruption details for older reports without changing estimates."""
    if not report or report.get('planned_rounds') is not None:
        return report
    view = dict(report, planned_rounds=config['request']['steps'])
    rows = []
    for row in report.get('trajectories', []):
        record = store.sample(run_id, row['sample_id'])
        rows.append(dict(row, planned_rounds=config['request']['steps'], recorded_rounds=len(row.get('rounds', [])),
                         error=record.get('error')))
    view['trajectories'] = rows
    return view


def build_question_report(run_id, config, records, error=None, *, masses=None):
    request = config["request"]
    contract = request["initial_world_state"]["benchmark_contract"]
    result = summarize_sequential(records, expected_steps=request["steps"])
    trajectories = [{key: value for key, value in row.items() if key != "branches"}
                    for row in result["trajectories"]]
    weighted = {}
    conditional = request.get('question_simulation', {}).get('algorithm') == 'conditional-stratified-v2'
    if masses is not None:
        yes_mass, no_mass = [], []
        for row, mass in zip(trajectories, masses, strict=True):
            outlook = (row["rounds"][-1].get("contract_outlook") or {}) if row["rounds"] else {}
            verdict = (row['p_yes'] if conditional else row["p_yes"] if row["p_yes"] in {0, 1}
                       and outlook.get("estimate_kind") == "verified_simulation" else None)
            row.update(represented_mass=mass, p_yes=verdict,
                       terminal_status="yes" if verdict == 1 else "no" if verdict == 0 else "estimated" if verdict is not None else "unresolved")
            if verdict is not None:
                yes_mass.append(mass * verdict)
                no_mass.append(mass * (1 - verdict))
            row.update(planned_rounds=request['steps'], recorded_rounds=len(row['rounds']))
        yes, no = math.fsum(yes_mass), math.fsum(no_mass)
        unknown = math.fsum(row["represented_mass"] for row in trajectories if row["p_yes"] is None)
        failed = result["failed_samples"] or result["invalid_trace_samples"]
        covered = math.fsum(yes_mass + no_mass)
        result.update(p_yes=min(1.0, yes) if unknown == 0 and trajectories else None,
                      covered_p_yes=yes / covered if conditional and covered > 0 else None,
                      unknown_bounds=[min(1.0, yes), min(1.0, yes + unknown)] if trajectories else None,
                      error="Some trajectories did not produce a valid decision trace." if failed else None)
        weighted = {"algorithm": request["question_simulation"]["algorithm"],
                    'aggregation_method':'weighted_conditional_probabilities' if conditional else 'verified_outcome_mass',
                    "probability_mass": {"yes": yes, "no": no, "unresolved": unknown}}
    known = sum(row["p_yes"] is not None for row in trajectories)
    drivers = {}
    final_bases = {}
    for trajectory in trajectories:
        if trajectory["p_yes"] is not None and trajectory["rounds"]:
            terminal = trajectory["rounds"][-1].get("contract_outlook") or {}
            if basis := terminal.get("basis"):
                final_bases[basis] = final_bases.get(basis, 0) + 1
        for round_ in trajectory["rounds"]:
            outlook = round_.get("contract_outlook") or {}
            if not outlook.get("change_driver") and outlook.get("estimate_kind") != "verified_simulation":
                continue
            key = " ".join((outlook.get("evidence_quote") or outlook["basis"]).split()).casefold()
            driver = drivers.setdefault(key, {
                "basis": outlook["basis"], "quote": outlook.get("evidence_quote", ""),
                "reference": outlook.get("evidence_ref"), "sample_ids": [],
            })
            if trajectory["sample_id"] not in driver["sample_ids"]:
                driver["sample_ids"].append(trajectory["sample_id"])
    return {
        "version": 3 if conditional else 2 if weighted else 1, "run_id": run_id, "question": contract["question"],
        'planned_rounds':request['steps'],
        **weighted,
        "status": "complete" if result["p_yes"] is not None else "preliminary" if known else "unavailable",
        "as_of": contract["as_of"], "outcome_deadline": contract["outcome_deadline"],
        "deadline_basis": contract.get("deadline_basis"), "criteria": contract["forecast_criteria"],
        "contract_rules": contract["rules"] if contract.get("market_url") else None,
        "market_url": contract.get("market_url"),
        "scheduled_milestones": contract.get("source_milestones", []),
        "p_yes": result["p_yes"], "conditional_estimate": result["covered_p_yes"],
        "bounds": result["unknown_bounds"], "estimated_trajectories": known,
        "planned_trajectories": len(trajectories), "knowledge_version": config.get("knowledge_version"),
        "drivers": sorted(drivers.values(), key=lambda row: -len(row["sample_ids"]))[:5],
        "final_bases": sorted(final_bases, key=final_bases.get, reverse=True)[:3] if not drivers else [],
        "sources": [{key: row[key] for key in ("title", "url", "revision_id", "revision_at") if key in row}
                    for row in contract.get("historical_evidence", [])],
        "trajectories": trajectories, "resume_available": result["resume_available"],
        "error": "The forecast stopped before the report could be completed." if error else result["error"],
        "warnings": [row["message"] for row in result["simulation_diagnostics"]["warnings"]],
        "verified_outcomes": sum((row["rounds"][-1].get("contract_outlook") or {}).get("estimate_kind") == "verified_simulation"
                                 for row in trajectories if row["p_yes"] is not None and row["rounds"]),
    }
